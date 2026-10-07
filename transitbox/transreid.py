"""Official TransReID ViT + JPM features for existing GHR-VLM person crops."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'models' / 'transreid' / 'vit_transreid_market.pth'
MODEL_VERSION = 'transreid-market-vit-jpm-v1'
DIMENSIONS = 3840
CHECKPOINT_URL = 'https://drive.google.com/file/d/11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO/view'


def checkpoint_sha256(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'TransReID trained checkpoint is missing: {path}. Run python tools/download_transreid.py')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_for_stride(stride):
    return SimpleNamespace(
        MODEL=SimpleNamespace(PRETRAIN_PATH='', PRETRAIN_CHOICE='none', COS_LAYER=False,
                              NECK='bnneck', TRANSFORMER_TYPE='vit_base_patch16_224_TransReID',
                              SIE_CAMERA=False, SIE_VIEW=False, SIE_COE=3.0, JPM=True,
                              STRIDE_SIZE=list(stride), DROP_PATH=0.1, ID_LOSS_TYPE='softmax',
                              SHUFFLE_GROUP=2, SHIFT_NUM=5, DEVIDE_LENGTH=4),
        INPUT=SimpleNamespace(SIZE_TRAIN=[256, 128]), TEST=SimpleNamespace(NECK_FEAT='before'))


def validate_state_dict(state):
    """Accept a trained ViT-B JPM checkpoint, never a plain ImageNet ViT."""
    required = ('base.patch_embed.proj.weight', 'base.pos_embed', 'classifier.weight',
                'b1.0.attn.qkv.weight', 'b2.0.attn.qkv.weight', 'bottleneck.running_mean')
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError('Expected a trained TransReID ViT + JPM checkpoint; missing ' + ', '.join(missing))
    if tuple(state['base.patch_embed.proj.weight'].shape) != (768, 3, 16, 16):
        raise ValueError('This adapter requires the official ViT-Base TransReID checkpoint')
    tokens = state['base.pos_embed'].shape[1]
    strides = [(stride, stride) for stride in (12, 14, 16)
               if ((256 - 16) // stride + 1) * ((128 - 16) // stride + 1) + 1 == tokens]
    if len(strides) != 1:
        raise ValueError(f'Unsupported TransReID positional grid: {tokens} tokens for 256x128 crops')
    return strides[0], int(state['classifier.weight'].shape[0])


def load_checkpoint(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a tensor state dictionary')
    for key in ('state_dict', 'model'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    stride, classes = validate_state_dict(state)
    return state, stride, classes


def build_model(state, stride, classes):
    from vendor.transreid.backbone import vit_base_patch16_224_TransReID
    from vendor.transreid.model import build_transformer_local
    model = build_transformer_local(classes, 0, 0, config_for_stride(stride),
                                    {'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID}, True)
    # Market1501 camera IDs have no correspondence to bus cameras. Neutral SIE
    # removes only the dataset camera embedding; every inference weight loads strictly.
    state = {key: value for key, value in state.items() if key != 'base.sie_embed'}
    model.load_state_dict(state, strict=True)
    return model.eval()


class TransReID:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device='auto', batch_size=8):
        import torch
        from torchvision import transforms
        self.checkpoint = Path(checkpoint)
        self.fingerprint = checkpoint_sha256(self.checkpoint)
        self.torch = torch
        torch.set_num_threads(8)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        state, stride, classes = load_checkpoint(self.checkpoint)
        self.model = build_model(state, stride, classes).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        self.description = {
            'model': 'TransReID ViT-Base + JPM', 'backbone': 'transreid',
            'repository': 'https://github.com/damo-cv/TransReID', 'dimensions': DIMENSIONS,
            'checkpoint': str(self.checkpoint), 'checkpointSha256': self.fingerprint,
            'trainingDataset': 'Market1501', 'stride': list(stride), 'device': str(self.device),
            'inputSize': [256, 128], 'pixelMean': [.5, .5, .5], 'pixelStd': [.5, .5, .5],
            'cameraEmbedding': 'disabled: no training-camera mapping for bus cameras',
            'clipPooling': 'mean of normalized frame features, then L2 normalize',
            'tracking': 'GHR-VLM only', 'samples': 8,
        }

    def extract(self, frames):
        from PIL import Image
        if not frames:
            raise ValueError('No passenger crops available')
        frames = [frames[int(i)] for i in np.unique(np.linspace(0, len(frames)-1, min(8, len(frames))).astype(int))]
        tensors = [self.transform(Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))) for frame in frames]
        features = []
        with self.torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                inputs = self.torch.stack(tensors[start:start+self.batch_size]).to(self.device)
                features.append(self.torch.nn.functional.normalize(self.model(inputs), dim=1))
            embedding = self.torch.nn.functional.normalize(self.torch.cat(features).mean(dim=0), dim=0)
        feature = embedding.cpu().numpy().astype(np.float32)
        if feature.shape != (DIMENSIONS,) or not np.isfinite(feature).all() or np.linalg.norm(feature) < .99:
            raise ValueError('Invalid TransReID clip embedding')
        return feature

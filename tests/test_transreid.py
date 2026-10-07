"""Adapter contract checks. Synthetic test features are never published as results."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torchvision import transforms

from transitbox.transreid import TransReID, DIMENSIONS, checkpoint_sha256, config_for_stride, validate_state_dict


class TransReIDTests(unittest.TestCase):
    def test_imagenet_weights_are_not_accepted_as_trained_reid(self):
        with self.assertRaisesRegex(ValueError, 'trained TransReID'):
            validate_state_dict({'patch_embed.proj.weight': torch.empty(768,3,16,16)})

    def test_checkpoint_grid_selects_matching_stride(self):
        for stride,tokens in ((12,211),(14,163),(16,129)):
            state={'base.patch_embed.proj.weight':torch.empty(768,3,16,16),
                   'base.pos_embed':torch.empty(1,tokens,768),'classifier.weight':torch.empty(751,768),
                   'b1.0.attn.qkv.weight':torch.empty(1),'b2.0.attn.qkv.weight':torch.empty(1),
                   'bottleneck.running_mean':torch.empty(768)}
            self.assertEqual(validate_state_dict(state),((stride,stride),751))

    def test_missing_weights_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError,'trained checkpoint is missing'):
                TransReID(Path(directory)/'missing.pth')

    def test_weight_fingerprint_changes_with_checkpoint_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pth';path.write_bytes(b'first test payload')
            first=checkpoint_sha256(path);path.write_bytes(b'second test payload')
            self.assertNotEqual(first,checkpoint_sha256(path))

    def test_rgb_normalization_and_frame_pooling(self):
        seen=[]
        def features(inputs):
            seen.extend(inputs.cpu())
            output=torch.zeros(len(inputs),DIMENSIONS)
            output[:,0]=inputs[:,0].mean(dim=(1,2))+2
            output[:,1]=inputs[:,2].mean(dim=(1,2))+2
            return output
        adapter=object.__new__(TransReID)
        adapter.torch=torch;adapter.device=torch.device('cpu');adapter.batch_size=1;adapter.model=features
        adapter.transform=transforms.Compose([
            transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
        red=np.full((40,20,3),[0,0,255],dtype=np.uint8)
        blue=np.full((40,20,3),[255,0,0],dtype=np.uint8)
        output=adapter.extract([red,blue])
        self.assertEqual(output.shape,(DIMENSIONS,));self.assertAlmostEqual(float(np.linalg.norm(output)),1,places=5)
        self.assertEqual(tuple(seen[0].shape),(3,256,128))
        self.assertTrue(torch.all(seen[0][0]==1));self.assertTrue(torch.all(seen[0][2]==-1))
        self.assertTrue(np.allclose(output[:2],np.array([1,1])/np.sqrt(2),atol=1e-6))

    def test_official_jpm_forward_is_3840_dimensional(self):
        # Small depth for an architecture smoke test; not a trained checkpoint.
        from vendor.transreid.backbone import TransReID as OfficialBackbone
        from vendor.transreid.model import build_transformer_local
        torch.set_num_threads(4)
        cfg=config_for_stride((16,16))
        def small_factory(**kwargs):
            return OfficialBackbone(patch_size=16,embed_dim=768,depth=2,num_heads=12,qkv_bias=True,**kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model=build_transformer_local(2,0,0,cfg,{'vit_base_patch16_224_TransReID':small_factory},True).eval()
        with torch.inference_mode():result=model(torch.zeros(1,3,256,128))
        self.assertEqual(tuple(result.shape),(1,DIMENSIONS))
        self.assertTrue(torch.isfinite(result).all())


if __name__=='__main__':unittest.main()

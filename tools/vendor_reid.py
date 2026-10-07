"""Copy only the BusProject inference code; never import its application entrypoint."""
from pathlib import Path
import ast, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'BusProject'
DEST=ROOT/'vendor'/'busproject'
files=[]
for relative in ['reid/models','Parsing/networks']:
    for source in (SOURCE/relative).glob('*.py'):
        target=DEST/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
        text=source.read_text()
        if source.name=='basebranch.py':text=text.replace('resnet50_s1(pretrained=True)','resnet50_s1(pretrained=False)')
        target.write_text(text);files.append(str(source.relative_to(SOURCE)))
for relative in ['reid/data/seqtransforms.py','Parsing/utils/transforms.py']:
    target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text((SOURCE/relative).read_text());files.append(relative)
for relative in ['reid/__init__.py','reid/data/__init__.py','Parsing/__init__.py','Parsing/utils/__init__.py']:
    (DEST/relative).write_text('')
# Inference-only implementation of the legacy InPlaceABN CUDA formula.
modules=DEST/'Parsing/modules';modules.mkdir(parents=True,exist_ok=True)
(modules/'__init__.py').write_text('''import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\nclass InPlaceABNSync(nn.Module):\n    def __init__(self,num_features,eps=1e-5,momentum=.1,affine=True,activation="leaky_relu",slope=.01):\n        super().__init__()\n        self.eps,self.activation,self.slope=eps,activation,slope\n        self.weight=nn.Parameter(torch.ones(num_features)) if affine else None\n        self.bias=nn.Parameter(torch.zeros(num_features)) if affine else None\n        self.register_buffer("running_mean",torch.zeros(num_features))\n        self.register_buffer("running_var",torch.ones(num_features))\n    def forward(self,x):\n        if self.training: raise RuntimeError("TransitBox ABN adapter supports inference only")\n        weight=self.weight.abs()+self.eps if self.weight is not None else None\n        x=F.batch_norm(x,self.running_mean,self.running_var,weight,self.bias,False,0,self.eps)\n        if self.activation=="leaky_relu": return F.leaky_relu(x,self.slope)\n        if self.activation=="relu": return F.relu(x)\n        if self.activation=="elu": return F.elu(x)\n        return x\n''')
source_text=(SOURCE/'__init__.py').read_text();tree=ast.parse(source_text)
names={'process_tracklet','featurize_img','FrameDataset','get_parsing','crop','grading','get_score','log_transformation','AdjustBackgroundTransform','weighted_average_filter'}
chunks=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dataset_settings' for t in node.targets):chunks.append(ast.get_source_segment(source_text,node))
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
        chunk=ast.get_source_segment(source_text,node)
        if node.name=='get_parsing':
            begin=chunk.index('    model = networks.init_model')
            end=chunk.index('    transform = transforms.Compose',begin)
            chunk=chunk[:begin]+'    model = parsing_model\n\n'+chunk[end:]
            chunk=chunk.replace('output = model(image.cuda())', 'output = model(image.to(device))')
            chunk=chunk.replace('os.environ["CUDA_VISIBLE_DEVICES"] = gpu','pass  # Device is selected by the worker, not changed per clip.')
        if node.name=='featurize_img':chunk=chunk.replace('tracklet = tracklet.squeeze(0)','tracklet = tracklet.squeeze(0).to(device)')
        chunks.append(chunk)
header='''"""BusProject inference functions, extracted without its tracking or startup side effects."""\nimport os,time,cv2\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom torch.utils.data import Dataset,DataLoader\nfrom torchvision import transforms\nfrom torchvision.transforms import functional as TF\nfrom PIL import Image\nfrom pytorch_msssim import ssim\nfrom reid.data import seqtransforms as T\nfrom Parsing.utils.transforms import get_affine_transform,transform_logits\n'''
(DEST/'feature_core.py').write_text(header+'\n\n'.join(chunks)+'\n')
(DEST/'PROVENANCE.json').write_text(json.dumps({'source':'../BusProject','files':files,'feature_core_functions':sorted(names),'source_sha256':hashlib.sha256(source_text.encode()).hexdigest(),'adaptations':['Avoid startup YOLO / filesystem cleanup','Reuse loaded parsing model','Move tracklet and parsing tensors to the selected CPU/CUDA device','Skip redundant ImageNet download before full checkpoint load','Evaluate legacy InPlaceABN formula using PyTorch ops; abs(weight)+eps retained']},indent=2))
print('Vendored BusProject inference only.')

"""Prepare evaluation-only TransReID modules from the official MIT source snapshot."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'transreid'


def prepare():
    backbone = (VENDOR / 'upstream' / 'vit_pytorch.py').read_text()
    model = (VENDOR / 'upstream' / 'make_model.py').read_text()
    compatibility = backbone.replace('from torch._six import container_abcs',
                                     'import collections.abc as container_abcs')
    (VENDOR / 'backbone.py').write_text(compatibility)
    names = {'shuffle_unit', 'weights_init_kaiming', 'weights_init_classifier', 'build_transformer_local'}
    segments = [ast.get_source_segment(model, node) for node in ast.parse(model).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(segments) != len(names):
        raise ValueError('Official TransReID source snapshot is incomplete')
    (VENDOR / 'model.py').write_text(
        '# Extracted unchanged from official TransReID make_model.py; see LICENSE.\n'
        'import copy\nimport torch\nimport torch.nn as nn\n\n' + '\n\n'.join(segments) + '\n')
    provenance = {
        'repository': 'https://github.com/damo-cv/TransReID',
        'license': 'MIT',
        'sources': {name: {'url': 'https://raw.githubusercontent.com/damo-cv/TransReID/main/model/' + relative,
                          'sha256': hashlib.sha256((VENDOR / 'upstream' / name).read_bytes()).hexdigest()}
                    for name, relative in [('vit_pytorch.py', 'backbones/vit_pytorch.py'), ('make_model.py', 'make_model.py')]},
        'changes': ['torch._six.container_abcs replaced with collections.abc for PyTorch 2.x',
                    'Only the original JPM evaluation class and initialization helpers are imported; training-only imports omitted'],
        'checkpoint_url': 'https://drive.google.com/file/d/11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO/view',
        'checkpoint_label': 'Official TransReID ViT Market1501 trained checkpoint',
    }
    (VENDOR / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    prepare()

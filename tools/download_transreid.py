"""Download the trained Market1501 checkpoint linked by the official TransReID GitHub."""
from pathlib import Path
import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transitbox.transreid import DEFAULT_CHECKPOINT, CHECKPOINT_URL, checkpoint_sha256, load_checkpoint


def download(destination=DEFAULT_CHECKPOINT, url=CHECKPOINT_URL, expected_sha256=None):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file():
        fingerprint=checkpoint_sha256(destination)
        if expected_sha256 and fingerprint.lower()!=expected_sha256.lower():
            raise ValueError('Existing checkpoint SHA256 does not match the requested checksum')
        load_checkpoint(destination)
        print(f'Validated existing trained checkpoint: {destination}',flush=True)
        return destination
    temporary=destination.with_suffix('.download')
    try:
        if 'drive.google.com' in url:
            packages=ROOT/'runtime/python-packages';sys.path.insert(0,str(packages))
            try:
                gdown=importlib.import_module('gdown')
            except ImportError:
                subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                                '--timeout','15','--retries','1','--target',str(packages),'gdown>=5.2,<6'],check=True)
                importlib.invalidate_caches();gdown=importlib.import_module('gdown')
            gdown.download(url=url,output=str(temporary),fuzzy=True,quiet=False,use_cookies=False)
        else:
            request=urllib.request.Request(url,headers={'User-Agent':'TransitBox/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as handle:
                for chunk in iter(lambda:response.read(1024*1024),b''):handle.write(chunk)
        fingerprint=checkpoint_sha256(temporary)
        if expected_sha256 and fingerprint.lower()!=expected_sha256.lower():
            raise ValueError('Downloaded checkpoint SHA256 does not match the requested checksum')
        _,stride,classes=load_checkpoint(temporary)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps({
            'repository':'https://github.com/damo-cv/TransReID','url':url,'sha256':fingerprint,
            'stride':list(stride),'classes':classes,'downloaded':True},indent=2)+'\n')
        print(f'Downloaded and validated TransReID checkpoint: {destination}',flush=True)
        return destination
    finally:
        if temporary.exists():temporary.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--url',default=CHECKPOINT_URL)
    parser.add_argument('--sha256')
    args=parser.parse_args()
    try:download(args.output,args.url,args.sha256)
    except Exception as error:
        print(f'TransReID download failed: {error}',file=sys.stderr);sys.exit(1)

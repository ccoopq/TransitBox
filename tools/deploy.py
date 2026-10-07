"""Download TransReID, recompute real features, refresh the page and serve TransitBox."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transitbox.transreid import DEFAULT_CHECKPOINT


def deploy(checkpoint=DEFAULT_CHECKPOINT,host='127.0.0.1',port=8787,device='auto',batch_size=8,skip_download=False):
    def run(script,*args):
        subprocess.run([sys.executable,str(ROOT/'tools'/script),*map(str,args)],cwd=ROOT,check=True)
    if not Path(checkpoint).is_file():
        if skip_download:raise FileNotFoundError(f'TransReID checkpoint is missing: {checkpoint}')
        run('download_transreid.py','--output',checkpoint)
    run('run_reid.py','--checkpoint',checkpoint,'--device',device,'--batch-size',batch_size)
    run('prepare_preview.py')
    print(f'Starting TransitBox at http://{host}:{port}',flush=True)
    subprocess.run([sys.executable,str(ROOT/'serve.py'),'--host',host,'--port',str(port)],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8787)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--skip-download',action='store_true')
    args=parser.parse_args()
    try:deploy(args.checkpoint,args.host,args.port,args.device,args.batch_size,args.skip_download)
    except (OSError,subprocess.CalledProcessError) as error:
        print(f'TransitBox deployment failed: {error}',file=sys.stderr);sys.exit(1)

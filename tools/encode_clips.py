"""Cache browser-compatible H.264 versions of GHR-VLM's MPEG-4 passenger clips."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, subprocess, shutil
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'GHR-VLM'

def ffmpeg_with_x264():
    candidates = list(dict.fromkeys(filter(None, [shutil.which('ffmpeg'), '/usr/bin/ffmpeg'])))
    for binary in candidates:
        if Path(binary).exists():
            result = subprocess.run([binary, '-hide_banner', '-encoders'], capture_output=True, text=True, check=True)
            if 'libx264 ' in result.stdout:
                return binary
    raise RuntimeError('An ffmpeg build with libx264 is required for browser-compatible clips.')

FFMPEG = ffmpeg_with_x264()

def encode(task):
    source, target = task
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return
    temporary = target.with_name(target.stem + '.encoding.mp4')
    subprocess.run([
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1',
        '-i', str(source), '-an', '-vf', 'scale=854:-2', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-crf', '26', '-threads', '1', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(temporary)
    ], check=True, stdout=subprocess.DEVNULL)
    temporary.replace(target)

if __name__ == '__main__':
    tasks = []
    for name in ('C3_1','C3_3'):
        output = SOURCE / f'outputs_{name}'
        for clip in json.loads((output / 'results.json').read_text()):
            basename = Path(clip['saved_clip_path']).name
            tasks.append((output / 'passenger_clips' / basename, ROOT / 'assets' / 'clips' / name / basename))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(encode, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 100 == 0 or i == len(futures):
                print(f'Encoded {i}/{len(futures)} clips', flush=True)

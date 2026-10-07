"""Check a static publication bundle without models, source datasets or network."""
from pathlib import Path
import argparse
import json
import re
from urllib.parse import urlparse


def check(output):
    output=Path(output).resolve();files=[p for p in output.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in output.rglob('*')):raise ValueError('Publication contains symlinks')
    ignored={'encoding.json','progress.txt','publication.json'}
    files=[p for p in files if p.name not in ignored]
    total=sum(p.stat().st_size for p in files)
    if total>=1_000_000_000:raise ValueError(f'GitHub Pages bundle exceeds 1 GB: {total} bytes')
    if any(p.stat().st_size>=100_000_000 for p in files):raise ValueError('A file exceeds the GitHub regular-file limit')
    html=(output/'index.html').read_text()
    match=re.search(r'<script id="transitbox-data"[^>]*>(.*?)</script>',html,re.S)
    if not match:raise ValueError('Missing embedded dataset')
    datasets=json.loads(match[1])
    manifest=json.loads((output/'runtime/reid_results.json').read_text())
    if manifest.get('backbone')!='transreid' or manifest.get('status')!='complete' or manifest.get('errors'):
        raise ValueError('The publication has no complete TransReID run')
    def exists(relative):
        if not relative:return
        if urlparse(relative).scheme:raise ValueError(f'Unexpected external media asset: {relative}')
        path=output/relative
        if output not in path.resolve().parents or not path.is_file():raise ValueError(f'Missing asset: {relative}')
    counts={};durations={}
    for name,data in datasets.items():
        exists(data['video']);exists(data['poster'])
        playlist=output/data['video'];text=playlist.read_text()
        info=json.loads((playlist.parent/'video.json').read_text())
        if abs(info['start_time'])>.025:raise ValueError(f'Web video begins at the wrong timestamp: {name}')
        if '#EXT-X-ENDLIST' not in text:raise ValueError(f'Incomplete web video: {name}')
        for reference in re.findall(r'URI="([^"]+)"',text):exists(str(playlist.parent.relative_to(output)/reference))
        for line in text.splitlines():
            if line and not line.startswith('#'):exists(str(playlist.parent.relative_to(output)/line))
        duration=sum(float(t) for t in re.findall(r'#EXTINF:([\d.]+)',text))
        if abs(duration-data['duration'])>5:raise ValueError(f'Web video timeline changed: {name}')
        durations[name]=round(duration,3);counts[name]=len(data['clips'])
        for clip in data['clips']:
            for key in ('clip','thumbnail','reidThumbnail','sheet'):exists(clip.get(key))
            result=manifest['results'].get(clip['id'])
            if not result or any(clip.get(k)!=result.get(k) for k in ('reid','reidStatus','matchedClip','reidPayment')):
                raise ValueError(f'Published identity differs from the real result: {clip["id"]}')
    if sum(counts.values())!=415:raise ValueError('Passenger clips are missing')
    return {'repository':'ccoopq/TransitBox','clips_by_source':counts,'clips':sum(counts.values()),
            'files':len(files),'bytes':total,'size_mib':round(total/1024**2,2),
            'video_duration_seconds':durations,'backbone':'transreid',
            'web_video':'HLS, 640x360, 6 fps, full original timelines',
            'server_inference':False,'checks':'All assets present; no symlinks; under size limits; real result agreement'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=Path('public'))
    args=parser.parse_args();print(json.dumps(check(args.output),indent=2))

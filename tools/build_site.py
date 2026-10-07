"""Build a portable GitHub Pages replay site from real, completed TransReID results."""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read_tag(html,name):
    match=re.search(r'<script id="'+re.escape(name)+r'"[^>]*>(.*?)</script>',html,re.S)
    if not match:raise ValueError(f'Missing embedded data: {name}')
    return json.loads(match[1])


def write_tag(html,name,value):
    encoded=json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    return re.sub(r'(<script id="'+re.escape(name)+r'"[^>]*>).*?(</script>)',
                  lambda match:match[1]+encoded+match[2],html,flags=re.S)


def build(output=None):
    from PIL import Image
    output=Path(output or ROOT/'public').resolve()
    if output==ROOT:raise ValueError('Publication output must not replace the project root')
    output.mkdir(parents=True,exist_ok=True)
    html=(ROOT/'index.html').read_text();datasets=read_tag(html,'transitbox-data')
    manifest=json.loads((ROOT/'runtime/reid_results.json').read_text())
    if manifest.get('backbone')!='transreid' or manifest.get('status')!='complete' or manifest.get('errors'):
        raise ValueError('Complete TransReID results are required before publishing')
    copied=set()
    def asset(relative):
        if not relative or relative in copied:return
        source=ROOT/relative;destination=output/relative
        if not source.is_file():raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.is_file() or destination.stat().st_size!=source.stat().st_size or destination.stat().st_mtime_ns!=source.stat().st_mtime_ns:
            shutil.copy2(source,destination)
        copied.add(relative)
    for name,data in datasets.items():
        playlist=output/'media'/name/'index.m3u8'
        if not playlist.is_file() or '#EXT-X-ENDLIST' not in playlist.read_text():
            raise ValueError(f'Encode the full web video first: python tools/encode_web_video.py {name}')
        video_info=output/'media'/name/'video.json'
        if not video_info.is_file() or abs(json.loads(video_info.read_text())['start_time'])>.025:
            raise ValueError(f'Web video time origin is not verified for {name}; run encode_web_video.py')
        data['video']=f'media/{name}/index.m3u8'
        data['webVideo']={'width':640,'height':360,'fps':6,'format':'HLS','timeline':'original'}
        asset(data['poster'])
        for clip in data['clips']:
            for key in ('clip','thumbnail','reidThumbnail'):asset(clip.get(key))
            sheet=clip.get('sheet')
            if sheet:
                source=ROOT/sheet;destination=output/sheet
                if not source.is_file():raise FileNotFoundError(source)
                destination.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(source) as image:
                    image.thumbnail((1280,1280))
                    image.convert('RGB').save(destination,quality=75,optimize=True)
    if manifest.get('model',{}).get('checkpoint'):
        manifest['model']['checkpoint']='models/transreid/'+Path(manifest['model']['checkpoint']).name
    metadata={key:value for key,value in manifest.items() if key!='results'}
    html=write_tag(html,'transitbox-data',datasets)
    html=write_tag(html,'transitbox-reid-meta',metadata)
    html=html.replace('Local playback workspace. No website has been published.',
                      'Playback workspace. Detection and payment results were computed before publication.')
    html=html.replace('Run python serve.py, then open http://127.0.0.1:8787.',
                      'Refresh the page or select another camera.')
    library='<script src="https://cdn.jsdelivr.net/npm/hls.js@1.6.13/dist/hls.min.js" crossorigin="anonymous"></script>\n'
    if library not in html:html=html.replace('<script>\n',library+'<script>\n',1)
    (output/'index.html').write_text(html)
    (output/'.nojekyll').write_text('')
    runtime=output/'runtime';runtime.mkdir(exist_ok=True)
    (runtime/'reid_results.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')))
    summary=json.loads((ROOT/'runtime/reid_summary.json').read_text())
    summary['model']=manifest['model']
    (runtime/'reid_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    from tools.check_publication import check
    report=check(output)
    (output/'publication.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'public')
    args=parser.parse_args();build(args.output)

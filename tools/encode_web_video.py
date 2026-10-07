"""Create a full-timeline HLS web copy; tracking/result timestamps stay unchanged."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.encode_clips import FFMPEG


def encode(source,output=None):
    target=Path(output or ROOT/'public'/'media'/source)
    target.mkdir(parents=True,exist_ok=True)
    video=ROOT/'media'/f'{source}.mp4'
    if not video.is_file():raise FileNotFoundError(video)
    signature={'path':str(video.resolve()),'size':video.stat().st_size,'mtime':video.stat().st_mtime_ns,
               'width':640,'fps':6,'crf':28,'maxrate':'180k','audio':'none','bframes':0,'version':2}
    cache=target/'encoding.json'
    if cache.is_file() and json.loads(cache.read_text()).get('signature')==signature and (target/'index.m3u8').is_file():
        print(f'{source}: reusing completed web video',flush=True);return
    playlist=target/'index.partial.m3u8'
    command=[FFMPEG,'-hide_banner','-loglevel','warning','-y','-i',str(video),
             '-map','0:v:0','-an','-vf','scale=640:360,setpts=PTS-STARTPTS,fps=6',
             '-c:v','libx264','-preset','veryfast','-crf','28','-maxrate','180k','-bufsize','360k',
             '-pix_fmt','yuv420p','-bf','0','-g','120','-keyint_min','120','-sc_threshold','0','-threads','4',
             '-f','hls','-hls_time','20','-hls_playlist_type','vod','-hls_segment_type','fmp4',
             '-hls_flags','independent_segments','-hls_fmp4_init_filename','init.mp4',
             '-hls_segment_filename',str(target/'segment_%05d.m4s'),
             '-progress',str(target/'progress.txt'),str(playlist)]
    print(f'{source}: encoding 640x360 full-timeline HLS',flush=True)
    subprocess.run(command,check=True,cwd=target)
    if '#EXT-X-ENDLIST' not in playlist.read_text():raise RuntimeError('Incomplete HLS playlist')
    playlist.replace(target/'index.m3u8')
    probe=subprocess.run([str(Path(FFMPEG).with_name('ffprobe')),'-v','error',
                          '-show_entries','format=start_time,duration','-of','json',str(target/'index.m3u8')],
                         capture_output=True,text=True,check=True)
    info=json.loads(probe.stdout)['format']
    if abs(float(info['start_time']))>.025:raise RuntimeError('Web video does not start at original time zero')
    (target/'video.json').write_text(json.dumps({'start_time':float(info['start_time']),
                'duration':float(info['duration']),'width':640,'height':360,'fps':6,'audio':False},indent=2)+'\n')
    cache.write_text(json.dumps({'signature':signature},indent=2)+'\n')
    print(f'{source}: HLS complete; {sum(p.stat().st_size for p in target.glob("*.m4s"))/1024**2:.1f} MiB',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',choices=['C3_1','C3_3'])
    args=parser.parse_args()
    encode(args.source)

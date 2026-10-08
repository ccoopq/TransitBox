"""Create a face-blurred website bundle without changing private originals."""
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import argparse
import bisect
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transitbox.privacy import FaceDetector, VERSION, MODEL_SHA256, MODEL_URL, head_box, redact, TemporalFaces, uncovered_heads
from tools.build_site import read_tag, write_tag
from tools.check_publication import check

PREFIX = 'redacted/faces-v1'
MEDIA_SUFFIXES = {'.mp4','.m4s','.m3u8','.jpg','.jpeg','.png','.webp'}
LOCAL = threading.local()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    result = subprocess.run(['/usr/bin/ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_frames:format=duration,start_time',
        '-of','json',str(path)], capture_output=True,text=True,check=True)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    rate = Fraction(stream['avg_frame_rate'])
    frames = stream.get('nb_frames')
    if not frames or frames == 'N/A':
        frames = round(float(data['format']['duration'])*float(rate))
    return {'width':stream['width'], 'height':stream['height'], 'rate':str(rate),
            'frames':int(frames), 'duration':float(data['format']['duration']),
            'start_time':float(data['format'].get('start_time',0))}


class Tracks:
    def __init__(self, data):
        self.frames = data['tracking']
        self.times = [f[0] for f in self.frames]

    def near(self, timestamp, width, height):
        start,end = bisect.bisect_left(self.times,timestamp-.45),bisect.bisect_right(self.times,timestamp+.45)
        return [head_box(obj[1:5],width,height) for frame in self.frames[start:end] for obj in frame[2]]

    def during(self, start, end, width, height):
        a,b = bisect.bisect_left(self.times,start-.45),bisect.bisect_right(self.times,end+.45)
        return [head_box(obj[1:5],width,height) for frame in self.frames[a:b] for obj in frame[2]]


def detector(model, device):
    if not hasattr(LOCAL,'detector'):
        LOCAL.detector = FaceDetector(model, device=device)
    return LOCAL.detector


def process_video(source, destination, context, offset, model, device, audit_dir, hls=False, batch_size=8):
    source,destination = Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    info = probe(source)
    width,height = info['width'],info['height']
    fps = float(Fraction(info['rate']))
    frame_bytes = width*height*3
    d = detector(model,device)
    tracks = Tracks(context)
    decode = ['/usr/bin/ffmpeg','-nostdin','-v','error','-threads','1','-i',str(source),
              '-an','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    encode = ['/usr/bin/ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24',
              '-s',f'{width}x{height}','-r',info['rate'],'-i','pipe:0','-an',
              '-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-threads','2','-bf','0']
    if hls:
        encode += ['-crf','28','-maxrate','180k','-bufsize','360k','-g','120','-keyint_min','120',
                   '-sc_threshold','0','-f','hls','-hls_time','20','-hls_playlist_type','vod',
                   '-hls_segment_type','fmp4','-hls_flags','independent_segments',
                   '-hls_fmp4_init_filename','init.mp4','-hls_segment_filename',
                   str(destination.parent/'segment_%05d.m4s'),str(destination)]
    else:
        encode += ['-crf','25','-movflags','+faststart',str(destination)]
    count,detected,protected_frames,masked_pixels = 0,0,0,0
    history = TemporalFaces()
    saved_stops = set()
    maximum_faces = 0
    started = last_log = time.monotonic()
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        reader = subprocess.Popen(decode,stdout=subprocess.PIPE,stderr=decode_log)
        writer = subprocess.Popen(encode,stdin=subprocess.PIPE,stderr=encode_log)
        try:
            while True:
                raw = reader.stdout.read(frame_bytes*batch_size)
                if not raw:
                    break
                if len(raw)%frame_bytes:
                    raise RuntimeError(f'Incomplete decoded frame: {source}')
                batch = np.frombuffer(raw,np.uint8).reshape(-1,height,width,3)
                predictions = d.detect_batch(list(batch))
                for image,faces in zip(batch,predictions):
                    timestamp = offset+count/fps
                    held = history.update(timestamp,faces)
                    heads = tracks.near(timestamp,width,height)
                    output,pixels = redact(image,held,uncovered_heads(held,heads))
                    writer.stdin.write(output.tobytes())
                    detected += len(faces)
                    masked_pixels += pixels
                    protected_frames += int(pixels>0)
                    if hls and faces:
                        stop = next((s for s in context['stops'] if s['start']<=timestamp<=s['end']),None)
                        label = None
                        if stop and stop['id'] not in saved_stops:
                            saved_stops.add(stop['id']);label = f"stop_{stop['id']:04d}"
                        if len(faces)>maximum_faces:
                            maximum_faces=len(faces);label='most_faces'
                        if label:
                            cv2.imwrite(str(audit_dir/(label+'_original.jpg')),image)
                            cv2.imwrite(str(audit_dir/(label+'_blurred.jpg')),output)
                            (audit_dir/(label+'.json')).write_text(json.dumps({'time':timestamp,'faces':len(faces)}))
                    count += 1
                if hls and time.monotonic()-last_log>=20:
                    elapsed=time.monotonic()-started
                    print(f'{context["name"]}: {count}/{info["frames"]} frames ({count/elapsed:.1f} fps), face boxes {detected}',flush=True)
                    last_log=time.monotonic()
            writer.stdin.close()
            reader.stdout.close()
            decode_status,encode_status = reader.wait(),writer.wait()
            if decode_status or encode_status:
                decode_log.seek(0);encode_log.seek(0)
                raise RuntimeError('Video processing failed: '+decode_log.read().decode()[-1500:]+encode_log.read().decode()[-1500:])
        except BaseException:
            reader.kill();writer.kill();reader.wait();writer.wait()
            raise
    if count!=info['frames']:
        raise ValueError(f'Frame count changed for {source}: {count} != {info["frames"]}')
    encoded = probe(destination)
    if encoded['frames']!=count or abs(encoded['duration']-info['duration'])>1/fps+.01 or abs(encoded['start_time'])>.025:
        raise ValueError(f'Output video timeline mismatch: {destination}')
    return {'frames':count,'detection_frames':count,'face_detections':detected,
            'protected_frames':protected_frames,'masked_pixels':masked_pixels,
            'width':width,'height':height,'fps':info['rate'],'duration':encoded['duration'],
            'elapsed_seconds':round(time.monotonic()-started,2)}


def rewrite_paths(value):
    if isinstance(value,str) and value.startswith(('assets/','media/')):
        return PREFIX+'/'+value
    if isinstance(value,list):
        return [rewrite_paths(item) for item in value]
    if isinstance(value,dict):
        return {key:rewrite_paths(item) for key,item in value.items()}
    return value


def run(source,output,model,device='cuda',workers=2):
    source,output = Path(source).resolve(),Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Use an independent output directory; originals are never overwritten')
    if (source/'privacy.json').exists():
        raise ValueError('Input already redacted; supply the original private website bundle')
    check(source)
    cv2.setNumThreads(1)
    output.mkdir(parents=True,exist_ok=True)
    state = output.parent/(output.name+'-state')
    state.mkdir(exist_ok=True)
    html = (source/'index.html').read_text()
    datasets = read_tag(html,'transitbox-data')
    context_hash = hashlib.sha256(json.dumps(datasets,sort_keys=True).encode()).hexdigest()
    results = {'sources':{},'clips':{},'images':{}}
    started = time.monotonic()

    def cached(key,signature,job):
        file = state/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        signature = {**signature,'version':VERSION,'model_sha256':MODEL_SHA256,'context_hash':context_hash}
        if file.exists():
            saved=json.loads(file.read_text())
            if saved['signature']==signature:
                return saved['result']
        result=job()
        temporary=file.with_suffix('.tmp')
        temporary.write_text(json.dumps({'signature':signature,'result':result}))
        temporary.replace(file)
        return result

    def video_job(name,data):
        path=source/data['video'];target=output/PREFIX/data['video']
        signature={'path':str(path),'size':path.stat().st_size,'mtime':path.stat().st_mtime_ns}
        result=cached('source:'+name,signature,lambda:process_video(path,target,data,0,model,device,state/name,hls=True))
        info=json.loads((path.parent/'video.json').read_text())
        info['privacy']=VERSION
        (target.parent/'video.json').write_text(json.dumps(info,indent=2)+'\n')
        return name,result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(video_job,name,data) for name,data in datasets.items()]
        for future in as_completed(futures):
            name,result=future.result();results['sources'][name]=result
            print(f'{name}: completed {result["frames"]} frames',flush=True)

    clips=[(name,data,clip) for name,data in datasets.items() for clip in data['clips']]
    def clip_job(item):
        name,data,clip=item;path=source/clip['clip'];target=output/PREFIX/clip['clip']
        signature={'sha256':digest(path)}
        return clip['id'],cached('clip:'+clip['id'],signature,
            lambda:process_video(path,target,data,clip['start'],model,device,state/'clip-audit'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(clip_job,item) for item in clips]):
            key,result=future.result();results['clips'][key]=result
            if len(results['clips'])%25==0:
                print(f'Passenger clips: {len(results["clips"])}/{len(clips)}',flush=True)

    image_context={}
    for name,data,clip in clips:
        for key in ('thumbnail','reidThumbnail','sheet'):
            if clip.get(key):image_context[clip[key]]=(data,clip)
    for name,data in datasets.items():image_context[data['poster']]=(data,None)
    images=[p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]

    def image_job(path):
        relative=str(path.relative_to(source));target=output/PREFIX/relative
        signature={'sha256':digest(path)}
        def process():
            image=cv2.imread(str(path));d=detector(model,device)
            if image is None:raise ValueError(f'Cannot read image: {path}')
            height,width=image.shape[:2];faces=[];heads=[]
            data,clip=image_context.get(relative,(None,None))
            if '_sheets/' in relative:
                for row in range(2):
                    for col in range(4):
                        x1,x2=col*width//4,(col+1)*width//4
                        y1,y2=row*height//2,(row+1)*height//2
                        for x,y,w,h in d.detect(image[y1:y2,x1:x2]):faces.append((x+x1,y+y1,w,h))
                        if data is not None and clip is not None:
                            for a,b,c,e in Tracks(data).during(clip['start'],clip['end'],x2-x1,y2-y1):
                                heads.append((a+x1,b+y1,c+x1,e+y1))
            else:
                faces=d.detect(image)
                if relative.startswith('assets/people/'):
                    heads=[(0,0,width,round(height*.55))]
                elif data is not None:
                    lookup=Tracks(data)
                    heads=lookup.during(clip['start'],clip['end'],width,height) if clip else lookup.near(data['initial'],width,height)
            blurred,pixels=redact(image,faces,uncovered_heads(faces,heads) if not relative.startswith('assets/people/') else heads)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(target),blurred,[cv2.IMWRITE_JPEG_QUALITY,85]):
                raise ValueError(f'Cannot write image: {target}')
            return {'face_detections':len(faces),'masked_pixels':pixels,'width':width,'height':height}
        return relative,cached('image:'+relative,signature,process)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(image_job,path) for path in images]):
            name,result=future.result();results['images'][name]=result
            if len(results['images'])%100==0:print(f'Images: {len(results["images"])}/{len(images)}',flush=True)

    html=write_tag(html,'transitbox-data',rewrite_paths(datasets))
    html=write_tag(html,'transitbox-reid-meta',rewrite_paths(read_tag(html,'transitbox-reid-meta')))
    (output/'index.html').write_text(html)
    (output/'.nojekyll').write_text('')
    (output/'runtime').mkdir(exist_ok=True)
    for path in (source/'runtime').glob('*.json'):
        (output/'runtime'/path.name).write_text(json.dumps(rewrite_paths(json.loads(path.read_text())),ensure_ascii=False,separators=(',',':')))
    report=check(output)
    (output/'publication.json').write_text(json.dumps(report,indent=2)+'\n')
    hashes={str(path.relative_to(output)):digest(path) for path in output.rglob('*') if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES}
    privacy={'status':'complete','version':VERSION,'detector':'OpenCV YuNet','model_url':MODEL_URL,
             'model_sha256':MODEL_SHA256,'device':device,'per_frame_detection':True,
             'temporal_hold_seconds':1.,'face_padding':.40,'asset_prefix':PREFIX,
             'sources':results['sources'],'clips_processed':len(results['clips']),
             'clip_frames':sum(item['frames'] for item in results['clips'].values()),
             'clip_detection_frames':sum(item['detection_frames'] for item in results['clips'].values()),
             'images_processed':len(results['images']),'media_sha256':hashes,
             'elapsed_seconds':round(time.monotonic()-started,2)}
    (output/'privacy.json').write_text(json.dumps(privacy,indent=2)+'\n')
    (state/'processing-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'status':'complete','output':str(output),'sources':results['sources'],
                      'clips':len(results['clips']),'images':len(results['images']),'size_mib':report['size_mib']},indent=2),flush=True)
    return privacy


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    run(args.source,args.output,args.model,args.device,args.workers)

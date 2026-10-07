"""Read the payment pipeline's existing passenger tracks; no detector is used here."""
from pathlib import Path
import csv
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def normalize_activity(value):
    return {'leaving_bus': 'alighting', 'staying_inside': 'inside'}.get(value, value or 'unknown')

def load_payment_map(output):
    """Map existing payment-only clips back to their original passenger clips."""
    refined_path=output/'results_2.json'
    payment_path=output/'_gpt_payment_review'/'gpt_payment_predictions.csv'
    if not refined_path.is_file() or not payment_path.is_file():return {}
    with payment_path.open() as handle:
        payments={row['clip']:row for row in csv.DictReader(handle)}
    return {Path(record.get('original_saved_clip_path',record['saved_clip_path'])).name:
            payments.get(Path(record['saved_clip_path']).name)
            for record in json.loads(refined_path.read_text())}

def load_tasks(source_root=None):
    source_root=Path(source_root or ROOT.parent/'GHR-VLM')
    tasks=[]
    for source in ('C3_1','C3_3'):
        output=source_root/f'outputs_{source}'
        events=json.loads((output/'flow_events.json').read_text())
        splits={e['stop_id']:e for e in events if e.get('type')=='passenger_split'}
        activity={Path(a['clip_path']).name:a for a in json.loads((output/'passenger_video_activity_prompt_test.json').read_text())}
        payments=load_payment_map(output)
        for record in json.loads((output/'results.json').read_text()):
            stop=splits[record['stop_id']]
            allowed={record['main_passenger_id']}
            for run in stop.get('id_run_segments',[]):
                if run.get('passenger_index')==record.get('passenger_index'):
                    allowed.update(run.get('main_passenger_ids',[]))
            frames=[]
            offsets=stop['frame_offsets']
            for frame in stop.get('passenger_tracker_debug',{}).get('frame_detections',[]):
                index=frame['frame_index']-1
                if not 0<=index<len(offsets):continue
                timestamp=stop['start_time']+offsets[index]
                if timestamp<record['clip_start_time']-.025 or timestamp>=record['clip_end_time']:continue
                objects=[o for o in frame.get('instances',[]) if o['object_id'] in allowed]
                if not objects:continue
                obj=max(objects,key=lambda o:(o['object_id']==record['main_passenger_id'],o.get('confidence',0)))
                bbox=obj['bbox']
                frames.append({'time':timestamp,'trackId':obj['object_id'],'box':[bbox[k] for k in ('x1_norm','y1_norm','x2_norm','y2_norm')],'confidence':obj.get('confidence',0)})
            filename=Path(record['saved_clip_path']).name
            tasks.append({'id':f"{source}-s{record['stop_id']:03d}-c{record['clip_id']:04d}",'source':source,'stop':record['stop_id'],'clipId':record['clip_id'],'trackId':record['main_passenger_id'],'trackIds':sorted(allowed),'start':record['clip_start_time'],'end':record['clip_end_time'],'activity':normalize_activity(activity.get(filename,{}).get('activity','unknown')),'path':str(output/'passenger_clips'/filename),'frames':frames})
            payment=payments.get(filename) or {}
            tasks[-1].update({'payment':payment.get('payment_type'),'paymentConfidence':payment.get('confidence')})
    # A result becomes available when its clip finishes, including overlapping clips.
    return sorted(tasks,key=lambda t:(t['source'],t['end'],t['start'],t['clipId']))

def crop_track(task,samples=8):
    detections=task['frames']
    if not detections:raise ValueError('Shared GHR-VLM track has no detections in this clip')
    indices=np.unique(np.linspace(0,len(detections)-1,min(samples,len(detections))).astype(int))
    capture=cv2.VideoCapture(task['path'])
    if not capture.isOpened():raise ValueError(f"Cannot open clip {task['id']}")
    fps=capture.get(cv2.CAP_PROP_FPS)
    crops=[];chosen=[]
    try:
        for index in indices:
            detection=detections[index]
            local=max(0,detection['time']-task['start'])
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(round(local*fps)))
            ok,frame=capture.read()
            if not ok:continue
            height,width=frame.shape[:2]
            x1,y1,x2,y2=detection['box']
            left,top=max(0,int(x1*width)),max(0,int(y1*height))
            right,bottom=min(width,int(np.ceil(x2*width))),min(height,int(np.ceil(y2*height)))
            if right-left<8 or bottom-top<8:continue
            crops.append(frame[top:bottom,left:right].copy());chosen.append(detection)
    finally:capture.release()
    if not crops:raise ValueError('No readable passenger crops from the shared track')
    return crops,chosen

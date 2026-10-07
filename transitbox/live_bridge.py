"""Adapt a newly saved GHR-VLM passenger clip to the exact existing ReID track format."""
from pathlib import Path
from transitbox.tracks import normalize_activity

def task_from_stream(record,split,source):
    allowed={record['main_passenger_id']}
    for run in split.get('id_run_segments',[]):
        if run.get('passenger_index')==record.get('passenger_index'):allowed.update(run.get('main_passenger_ids',[]))
    offsets=split['frame_offsets'];frames=[]
    for frame in split.get('passenger_tracker_debug',{}).get('frame_detections',[]):
        i=frame['frame_index']-1
        if not 0<=i<len(offsets):continue
        timestamp=split['start_time']+offsets[i]
        if not record['clip_start_time']-.025<=timestamp<record['clip_end_time']:continue
        detections=[o for o in frame.get('instances',[]) if o['object_id'] in allowed]
        if not detections:continue
        obj=max(detections,key=lambda o:(o['object_id']==record['main_passenger_id'],o.get('confidence',0)))
        frames.append({'time':timestamp,'trackId':obj['object_id'],'box':[obj['bbox'][k] for k in ('x1_norm','y1_norm','x2_norm','y2_norm')],'confidence':obj.get('confidence',0)})
    path=Path(record['saved_clip_path']).resolve()
    return {'id':f"{source}-s{record['stop_id']:03d}-c{record['clip_id']:04d}",'source':source,'stop':record['stop_id'],'clipId':record['clip_id'],'trackId':record['main_passenger_id'],'trackIds':sorted(allowed),'start':record['clip_start_time'],'end':record['clip_end_time'],'activity':normalize_activity(record.get('passenger_activity')),'payment':record.get('payment_type'),'paymentConfidence':record.get('payment_confidence'),'path':str(path),'frames':frames}

"""Consume GHR-VLM clip tasks on stdin in a separate inference process."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transitbox.tracks import crop_track
from transitbox.matching import IdentityGallery
from tools.run_reid import atomic_json


def replay_gallery(tasks, features, metadata):
    """Replay available features in video order after a late activity label.

    Activity updates use existing embeddings, never another model/VLM call.
    Replaying also keeps an earlier exit from seeing later boarding entries.
    """
    gallery=IdentityGallery();results={}
    for task in sorted(tasks.values(),key=lambda t:(t['source'],t['end'],t['start'],t['clipId'])):
        info=metadata[task['id']]
        if task['id'] in features:
            result=gallery.associate(task,features[task['id']]);result.update(info)
        else:
            result={'reid':None,'reidStatus':'error',**info}
        results[task['id']]=result
    return results,{source:gallery.size(source) for source in {t['source'] for t in tasks.values()}}

if __name__=='__main__':
    # A live run has its own result file, keeping completed local replay results intact.
    model=None;tasks={};features={};metadata={};output=ROOT/'runtime/live_reid_results.json'
    manifest={'results':{},'total':0,'completed':0,'errors':0,'status':'running','trackingRecomputed':False,
              'backbone':'transreid','matchingVersion':'boarding-gallery-v1','threshold':None,'updated':time.time()}
    for line in sys.stdin:
        message=json.loads(line)
        if message.get('type')=='finish':break
        if message.get('type')=='activity':
            task=tasks[message['id']]
            task['activity']=message['activity']
            for key in ('payment','paymentConfidence'):
                if key in message:task[key]=message[key]
        else:
            task=message['task'];tasks[task['id']]=task
            info={'start':task['start'],'end':task['end'],'source':task['source'],'stop':task['stop'],
                  'trackId':task['trackId'],'trackIds':task['trackIds'],'error':None}
            try:
                if model is None:
                    from transitbox.reid import TransReID
                    model=TransReID();manifest['model']=model.description
                crops,selected=crop_track(task);features[task['id']]=model.extract(crops)
                info['samples']=len(selected)
            except Exception as error:
                features.pop(task['id'],None);info['error']=str(error)
            metadata[task['id']]=info
        manifest['results'],manifest['onboardBySource']=replay_gallery(tasks,features,metadata)
        manifest['total']=manifest['completed']=len(tasks)
        manifest['errors']=sum(r['reidStatus']=='error' for r in manifest['results'].values())
        manifest['awaitingActivity']=sum(r['reidStatus']=='awaiting_activity' for r in manifest['results'].values())
        manifest['updated']=time.time();atomic_json(output,manifest)
    manifest['status']='complete_with_errors' if manifest['errors'] else 'awaiting_activity' if manifest.get('awaitingActivity') else 'complete'
    manifest['updated']=time.time();atomic_json(output,manifest)

"""Run official TransReID on the GHR-VLM payment task's passenger track list."""
from pathlib import Path
import sys,json,time,argparse,hashlib
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transitbox.tracks import load_tasks,crop_track
from transitbox.matching import IdentityGallery
from transitbox.transreid import DEFAULT_CHECKPOINT, MODEL_VERSION, DIMENSIONS, checkpoint_sha256


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')))
    temporary.replace(path)


def run(limit=0,threshold=None,cached_only=False,checkpoint=DEFAULT_CHECKPOINT,device='auto',batch_size=8):
    import numpy as np
    tasks=load_tasks();tasks=tasks[:limit] if limit else tasks
    output=ROOT/'runtime'
    results={};gallery=IdentityGallery(threshold)
    previous_path=output/'reid_results.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous.get('backbone')!='transreid':
        history=output/'history';history.mkdir(exist_ok=True)
        for name in ('reid_results.json','reid_summary.json'):
            path=output/name
            backup=history/f'pre_transreid_{name}'
            if path.exists() and not backup.exists():backup.write_bytes(path.read_bytes())
    # Keep the extraction version stable: matching changes do not invalidate features.
    manifest={'trackingSource':'GHR-VLM payment passenger_tracker_debug.frame_detections',
              'trackingRecomputed':False,'vlmRecomputed':False,'paymentRecomputed':False,
              'total':len(tasks),'completed':0,'errors':0,'status':'running','results':results,
              'threshold':threshold,'version':MODEL_VERSION,'backbone':'transreid','matchingVersion':'boarding-gallery-v1',
              'matchingPolicy':'top-1 cosine among earlier onboard boarding identities; remove on match',
              'model':{'model':'TransReID ViT-Base + JPM','backbone':'transreid','dimensions':DIMENSIONS,
                       'checkpoint':str(checkpoint),'device':device},
              'updated':time.time()}
    def save():manifest['updated']=time.time();atomic_json(output/'reid_results.json',manifest)
    try:fingerprint=checkpoint_sha256(checkpoint)
    except FileNotFoundError as error:
        manifest.update(status='awaiting_weights',error=str(error),onboardBySource={'C3_1':0,'C3_3':0},cacheHits=0,featuresExtracted=0)
        save()
        atomic_json(output/'reid_summary.json',{'backbone':'transreid','status':'awaiting_weights','clips':len(tasks),'reid_completed':0,
                    'error':str(error),'tracking_recomputed':False,'vlm_recomputed':False,'payment_recomputed':False})
        raise
    manifest['model']['checkpointSha256']=fingerprint
    cache=output/'embeddings'/MODEL_VERSION/fingerprint[:16];cache.mkdir(parents=True,exist_ok=True)
    save();model=None
    if not cached_only:
        try:
            from transitbox.reid import TransReID
            model=TransReID(checkpoint=checkpoint,device=device,batch_size=batch_size)
            manifest['model']=model.description;save()
        except Exception as error:
            manifest.update(status='model_error',error=str(error),errors=1,onboardBySource={'C3_1':0,'C3_3':0})
            save();raise RuntimeError(f'TransReID checkpoint could not be loaded: {error}') from error
    for i,task in enumerate(tasks,1):
        before=time.perf_counter()
        try:
            cache_path=cache/f"{task['id']}.npz"
            signature=hashlib.sha256(json.dumps({'version':manifest['version'],'checkpointSha256':fingerprint,'path':task['path'],'mtime':Path(task['path']).stat().st_mtime_ns,'frames':task['frames']},sort_keys=True).encode()).hexdigest()
            loaded=False
            if cache_path.exists():
                with np.load(cache_path,allow_pickle=False) as previous:
                    if str(previous['signature'])==signature:
                        feature=previous['feature'];selected=json.loads(str(previous['selected']))
                        loaded=feature.shape==(DIMENSIONS,) and np.isfinite(feature).all() and abs(float(np.linalg.norm(feature))-1)<1e-5
            if not loaded:
                if cached_only:raise ValueError('Missing or stale embedding cache; cached-only run cannot extract features')
                if model is None:
                    from transitbox.reid import TransReID
                    model=TransReID(checkpoint=checkpoint,device=device,batch_size=batch_size);manifest['model']=model.description
                crops,selected=crop_track(task);feature=model.extract(crops)
                np.savez_compressed(cache_path,feature=feature,signature=signature,selected=json.dumps(selected))
                if crops:
                    import cv2
                    thumbnail=ROOT/'assets'/'people'/f"{task['id']}.jpg";thumbnail.parent.mkdir(parents=True,exist_ok=True)
                    crop=crops[len(crops)//2];h,w=crop.shape[:2];scale=160/max(h,w);cv2.imwrite(str(thumbnail),cv2.resize(crop,(max(1,int(w*scale)),max(1,int(h*scale)))))
            result=gallery.associate(task,feature)
            result.update({'source':task['source'],'stop':task['stop'],'start':task['start'],'end':task['end'],
                           'trackId':task['trackId'],'trackIds':task['trackIds'],
                           'payment':task.get('payment'),'paymentConfidence':task.get('paymentConfidence'),
                           'samples':len(selected),'sampleTimes':[round(d['time'],3) for d in selected],
                           'elapsed':round(time.perf_counter()-before,3),'thumbnail':f"assets/people/{task['id']}.jpg",'embeddingCached':loaded,'error':None})
        except Exception as error:
            manifest['errors']+=1;result={'reid':None,'reidStatus':'error','error':str(error),'source':task['source'],'stop':task['stop'],'trackId':task['trackId']}
        results[task['id']]=result;manifest['completed']=i;save()
        if i%10==0 or i==len(tasks):print(f'ReID {i}/{len(tasks)}; errors={manifest["errors"]}; last={result.get("reidStatus")}',flush=True)
    manifest['status']='complete' if not manifest['errors'] else 'complete_with_errors'
    manifest['onboardBySource']={source:gallery.size(source) for source in ('C3_1','C3_3')}
    manifest['cacheHits']=sum(r.get('embeddingCached',False) for r in results.values())
    manifest['featuresExtracted']=sum(r.get('embeddingCached') is False for r in results.values())
    save()
    scores=[r['reidSimilarity'] for r in results.values() if r['reidStatus']=='matched']
    summary={'clips':len(tasks),'reid_completed':manifest['completed'],'errors':manifest['errors'],
             'backbone':'transreid','model':manifest['model'],
             'payment_classifications':sum(bool(t.get('payment')) for t in tasks),
             'boarding_identities':gallery.counter,'association_counts':dict(Counter(r['reidStatus'] for r in results.values())),
             'activity_counts':dict(Counter(t['activity'] for t in tasks)),
             'onboard_by_source':manifest['onboardBySource'],'cosine_threshold':threshold,
             'matching_policy':manifest['matchingPolicy'],'matching_version':manifest['matchingVersion'],
             'cache_hits':manifest['cacheHits'],'features_extracted':manifest['featuresExtracted'],
             'tracking_recomputed':False,'vlm_recomputed':False,'payment_recomputed':False,
             'matched_similarity':{'min':min(scores),'max':max(scores),'mean':round(sum(scores)/len(scores),5)} if scores else None}
    atomic_json(output/'reid_summary.json',summary)
    print('Saved actual model results to runtime/reid_results.json',flush=True)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--threshold',type=float,default=None,help='Optional cosine cutoff; default accepts the nearest eligible boarding identity')
    parser.add_argument('--cached-only',action='store_true',help='Rebuild gallery using cached features; never load a model or invoke VLM')
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device',default='auto',help='auto, cpu, or cuda:0')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    try:result=run(args.limit,args.threshold,args.cached_only,args.checkpoint,args.device,args.batch_size)
    except (FileNotFoundError,ValueError,RuntimeError) as error:
        print(f'TransReID could not start: {error}',file=sys.stderr);sys.exit(1)
    if result['errors']:sys.exit(1)

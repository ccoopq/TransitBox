"""Run the original GHR-VLM tracker and stream its clips to a separate TransReID worker.

Use the GHR-VLM environment for this entrypoint. Arguments after '--' are passed
unchanged to GHR-VLM, including input video, tracking models, time range and output directory.
This command performs NEW tracking; the default local preview uses already saved tracks.
"""
from pathlib import Path
import argparse,sys,subprocess,importlib.util,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transitbox.live_bridge import task_from_stream

if __name__=='__main__':
    own=argparse.ArgumentParser(description=__doc__)
    own.add_argument('--source',choices=['C3_1','C3_3'],required=True)
    own.add_argument('--reid-python',default='/home/ikun/miniconda3/bin/python')
    own.add_argument('--activity-results',type=Path,help='Existing activity JSON for these exact clips; avoids new VLM calls')
    own_args,ghr_args=own.parse_known_args();ghr_args=[a for a in ghr_args if a!='--']
    spec=importlib.util.spec_from_file_location('transitbox_ghr_stream',ROOT/'vendor/ghr_vlm/stream.py')
    stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
    sys.argv=[str(ROOT/'vendor/ghr_vlm/stream.py'),*ghr_args]
    ghr_config=stream.parse_args()
    activities={}
    if own_args.activity_results:
        activities={Path(row['clip_path']).name:row['activity'] for row in json.loads(own_args.activity_results.read_text())}
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'live_reid_worker.log').open('w') as log:
        worker=subprocess.Popen([own_args.reid_python,str(ROOT/'tools/reid_worker.py')],stdin=subprocess.PIPE,stdout=log,stderr=log,text=True,bufsize=1,cwd=ROOT)
        def clip_ready(record,split):
            if worker.poll() is not None:raise RuntimeError('ReID worker stopped; see runtime/live_reid_worker.log')
            task=task_from_stream(record,split,own_args.source)
            if task['activity']=='unknown' and Path(task['path']).name in activities:
                from transitbox.tracks import normalize_activity
                task['activity']=normalize_activity(activities[Path(task['path']).name])
            worker.stdin.write(json.dumps({'task':task})+'\n');worker.stdin.flush()
        try:
            stream.process_streaming_video(ghr_config,clip_callback=clip_ready)
            worker.stdin.write('{"type":"finish"}\n');worker.stdin.flush();worker.stdin.close()
            if worker.wait()!=0:raise RuntimeError('ReID worker failed; see runtime/live_reid_worker.log')
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()

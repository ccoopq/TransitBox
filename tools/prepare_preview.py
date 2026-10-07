"""Build the local HTML data snapshot from existing GHR-VLM outputs; no inference."""
from pathlib import Path
import csv, json, bisect
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transitbox.tracks import normalize_activity
import cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'GHR-VLM'

def relative_asset(path):
    return str(path.relative_to(ROOT))

def build():
    datasets = {}
    reid_path = ROOT / 'runtime' / 'reid_results.json'
    reid_manifest = json.loads(reid_path.read_text()) if reid_path.exists() else {'results': {}}
    # Never display archived BusProject assignments as TransReID output.
    if reid_manifest.get('backbone') != 'transreid':
        reid_manifest={'backbone':'transreid','status':'awaiting_weights','completed':0,'results':{}}
    for name in ('C3_1', 'C3_3'):
        output = SOURCE / f'outputs_{name}'
        records = json.loads((output / 'results.json').read_text())
        refined = json.loads((output / 'results_2.json').read_text())
        activities = json.loads((output / 'passenger_video_activity_prompt_test.json').read_text())
        activity_map = {Path(x['clip_path']).name: x for x in activities}
        payment_rows = list(csv.DictReader((output / '_gpt_payment_review/gpt_payment_predictions.csv').open()))
        payment_map = {x['clip']: x for x in payment_rows}
        refinement_map = {Path(x.get('original_saved_clip_path', x['saved_clip_path'])).name: x for x in refined}
        flow = json.loads((output / 'flow_events.json').read_text())
        splits = [x for x in flow if x['type'] == 'passenger_split']
        tracking = []
        for stop in splits:
            debug = stop.get('passenger_tracker_debug', {})
            offsets = stop.get('frame_offsets', [])
            for frame in debug.get('frame_detections', []):
                index = frame['frame_index'] - 1
                if index >= len(offsets):
                    continue
                tracking.append([round(stop['start_time'] + offsets[index], 3), stop['stop_id'], [
                    [obj['object_id'], *[round(obj['bbox'][k], 5) for k in ('x1_norm','y1_norm','x2_norm','y2_norm')], round(obj.get('confidence', 0), 3)]
                    for obj in frame.get('instances', [])
                ]])
        tracking.sort(key=lambda frame: frame[0])
        clips = []
        thumb_dir = ROOT / 'assets' / name
        thumb_dir.mkdir(parents=True, exist_ok=True)
        for r in records:
            basename = Path(r['saved_clip_path']).name
            refined_record = refinement_map.get(basename)
            p = payment_map.get(Path(refined_record['saved_clip_path']).name) if refined_record else None
            a = activity_map.get(basename, {})
            sheet = output / 'passenger_clip_sheets' / Path(r.get('passenger_sheet_path') or '').name
            thumbnail = thumb_dir / f"clip_{r['clip_id']:04d}.jpg"
            if sheet.is_file():
                with Image.open(sheet) as image:
                    # The upstream sheet uses four columns and two rows, including labels.
                    tile = image.crop((0, 0, image.width // 4, image.height // 2))
                    tile.thumbnail((240, 160))
                    tile.save(thumbnail, quality=80)
            clips.append({
                'id': f"{name}-s{r['stop_id']:03d}-c{r['clip_id']:04d}",
                'clipId': r['clip_id'], 'stop': r['stop_id'], 'trackId': r.get('main_passenger_id'),
                'start': r['clip_start_time'], 'end': r['clip_end_time'], 'duration': r['clip_duration'],
                'activity': normalize_activity(a.get('activity', 'unknown')), 'activityConfidence': a.get('activity_confidence'),
                'payment': p['payment_type'] if p else None, 'paymentConfidence': p['confidence'] if p else None,
                'reason': p['reasoning'] if p else None, 'evidence': p['evidence_frames'] if p else None,
                'reid': None, 'reidStatus': 'awaiting_model' if reid_manifest.get('status') in ('awaiting_weights','model_error') else 'pending',
                'thumbnail': relative_asset(thumbnail) if thumbnail.exists() else None,
                'clip': f'assets/clips/{name}/{basename}',
                'sheet': f'media/{name}_sheets/{sheet.name}' if sheet.exists() else None,
            })
        for clip in clips:
            result = reid_manifest['results'].get(clip['id'])
            if result:
                clip.update({key: result.get(key) for key in (
                    'reid', 'reidStatus', 'reidSimilarity', 'matchedClip', 'candidates',
                    'reidBoardingClip', 'reidBoardingStop', 'reidPayment', 'reidPaymentConfidence',
                    'galleryBefore', 'galleryAfter', 'galleryAction', 'unmatchedReason', 'repeatedClip', 'trackIds')})
                clip['reidThumbnail'] = result.get('thumbnail')
                clip['reidError'] = result.get('error')
        clips.sort(key=lambda r: r['start'])
        first = clips[0]['start']
        target = min(first + 20.0, splits[0]['end_time'] - 1)
        candidates = [f for f in tracking if first + 10 <= f[0] <= first + 30 and f[2]]
        initial = min(candidates, key=lambda f: abs(f[0] - target))[0] + 0.04 if candidates else target
        capture = cv2.VideoCapture(str(SOURCE / 'dataset' / f'{name}.mp4'))
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        capture.set(cv2.CAP_PROP_POS_MSEC, initial * 1000)
        ok, frame = capture.read()
        capture.release()
        if ok:
            cv2.imwrite(str(ROOT / 'assets' / f'{name}_poster.jpg'), frame)
        roi = json.loads((output / 'payment_roi_detected.json').read_text())
        datasets[name] = {
            'name': name, 'video': f'media/{name}.mp4', 'poster': f'assets/{name}_poster.jpg',
            'initial': initial, 'duration': duration, 'clips': clips, 'tracking': tracking,
            'width': 1280, 'height': 720,
            'paymentROI': [round(x / (1280 if i % 2 == 0 else 720), 5) for i, x in enumerate(roi['bbox_original_pixels'])],
            'stops': [{'id': s['stop_id'], 'start': s['start_time'], 'end': s['end_time']} for s in splits],
        }
        links = {
            ROOT / 'media' / f'{name}.mp4': SOURCE / 'dataset' / f'{name}.mp4',
            ROOT / 'media' / f'{name}_clips': output / 'passenger_clips',
            ROOT / 'media' / f'{name}_sheets': output / 'passenger_clip_sheets',
        }
        for dest, source in links.items():
            if dest.is_symlink():
                dest.unlink()
            if not dest.exists():
                dest.symlink_to(source, target_is_directory=source.is_dir())
        print(f'{name}: {len(clips)} clips, {sum(bool(c["payment"]) for c in clips)} payments, {len(tracking)} tracking frames')
    html_path = ROOT / 'index.html'
    html = html_path.read_text()
    start = '<script id="transitbox-data" type="application/json">'
    end = '</script><!-- /transitbox-data -->'
    a, rest = html.split(start, 1)
    _, b = rest.split(end, 1)
    blob = json.dumps(datasets, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = a + start + blob + end + b
    import re
    counts = {status: sum(r.get('reidStatus') == status for r in reid_manifest['results'].values())
              for status in ('boarded', 'matched', 'unmatched')}
    summary = (f"{reid_manifest.get('completed', 0)} clips processed with TransReID. "
               f"{counts['boarded']} boarding identities, {counts['matched']} nearest exit matches, "
               f"{counts['unmatched']} unmatched exits. Existing activity and payment results are reused.")
    if reid_manifest.get('status') in ('awaiting_weights','model_error'):
        summary='TransReID is selected but its trained checkpoint is unavailable. No TransReID identities have been computed. Tracking, activity and payment results are available.'
    html = re.sub(r'(<span id="reid-run-summary">).*?(</span>)',
                  lambda match: match[1] + summary + match[2], html)
    metadata={key:value for key,value in reid_manifest.items() if key!='results'}
    tag='<script id="transitbox-reid-meta" type="application/json">'+json.dumps(metadata,separators=(',',':')).replace('</','<\\/')+'</script>'
    if '<script id="transitbox-reid-meta"' in html:
        html=re.sub(r'<script id="transitbox-reid-meta"[^>]*>.*?</script>',lambda _:tag,html)
    else:html=html.replace(end,end+'\n'+tag)
    html_path.write_text(html)
    print('Embedded a sanitized, local snapshot into index.html.')

if __name__ == '__main__':
    build()

"""Reject unredacted assets and verify the complete published media inventory."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from transitbox.privacy_constants import MODEL_SHA256, VERSION

MEDIA_SUFFIXES = {'.mp4','.m4s','.m3u8','.jpg','.jpeg','.png','.webp'}


def check(root):
    root = Path(root).resolve()
    report_path = root/'privacy.json'
    if not report_path.is_file():
        raise ValueError('Face-redaction report missing; refusing to publish original media')
    report = json.loads(report_path.read_text())
    if report.get('status')!='complete' or report.get('version')!=VERSION or report.get('model_sha256')!=MODEL_SHA256:
        raise ValueError('Face-redaction report is incomplete or uses an unexpected detector')
    if not report.get('per_frame_detection'):
        raise ValueError('Video redaction must detect faces on every frame')
    html = (root/'index.html').read_text()
    match = re.search(r'<script id="transitbox-data"[^>]*>(.*?)</script>',html,re.S)
    data = json.loads(match[1])
    if set(report['sources'])!=set(data):
        raise ValueError('A source video has not been redacted')
    for name,source in report['sources'].items():
        if source['frames']<=0 or source['detection_frames']!=source['frames'] or source['face_detections']<=0:
            raise ValueError(f'Incomplete per-frame face processing: {name}')
        playlist = (root/data[name]['video']).read_text()
        duration = sum(float(value) for value in re.findall(r'#EXTINF:([\d.]+)',playlist))
        if round(duration*float(Fraction(source['fps'])))!=source['frames']:
            raise ValueError(f'Redacted video frame count does not match the timeline: {name}')
    if report['clips_processed']!=sum(len(d['clips']) for d in data.values()):
        raise ValueError('Passenger clip redaction is incomplete')
    if report['clip_detection_frames']!=report['clip_frames'] or report['clip_frames']<=0:
        raise ValueError('Passenger clips must also detect faces on every frame')
    media = {str(p.relative_to(root)):p for p in root.rglob('*')
             if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES}
    if set(media)!=set(report['media_sha256']):
        raise ValueError('Published media differs from the redacted inventory')
    image_count = sum(path.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} for path in media.values())
    if image_count!=report['images_processed']:
        raise ValueError('Screenshots and passenger crops were not all processed')
    prefix = report['asset_prefix']+'/'
    for name,path in media.items():
        if not name.startswith(prefix) or path.is_symlink():
            raise ValueError(f'Original/unversioned media must not be deployed: {name}')
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(4*1024*1024),b''):
                h.update(chunk)
        if h.hexdigest()!=report['media_sha256'][name]:
            raise ValueError(f'Media is not the verified redacted file: {name}')
    print(f'PASS: redacted {len(report["sources"])} source videos, {report["clips_processed"]} clips, {image_count} images; complete media hashes verified')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    check(args.root)

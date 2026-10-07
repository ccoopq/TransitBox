"""Local GHR-VLM streaming source with a clip-ready callback and no embedded keys."""
from pathlib import Path
import ast,re,json,hashlib
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'GHR-VLM';DEST=ROOT/'vendor/ghr_vlm';DEST.mkdir(parents=True,exist_ok=True)
files=['stream.py','sam3_door_detector.py','gpt_payment_classifier.py']
manifest={'source':'../GHR-VLM','files':{},'changes':['Removed embedded API credentials','Added a clip-ready callback after the original tracking and segmentation']}
for name in files:
    source=(SOURCE/name).read_text();tree=ast.parse(source);lines=source.splitlines(keepends=True)
    # Blank credential assignments, preserving the environment-based configuration.
    edits=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            sensitive=node.value.value.startswith(('sk-','sk_')) or any(isinstance(t,ast.Name) and ('API_KEY' in t.id or 'API_TOKEN' in t.id) for t in node.targets)
            if sensitive:
                names=[t.id for t in node.targets if isinstance(t,ast.Name)]
                if names:edits.append((node.lineno-1,node.end_lineno,','.join(names)+' = ""  # Configure credentials using environment variables.\n'))
    for start,end,replacement in sorted(edits,reverse=True):lines[start:end]=[replacement]
    text=''.join(lines)
    if name=='stream.py':
        text=text.replace('def process_streaming_video(args: argparse.Namespace) -> List[Dict[str, Any]]:', 'def process_streaming_video(args: argparse.Namespace, clip_callback=None) -> List[Dict[str, Any]]:')
        needle='            results.append(result)\n            write_results(results, output_dir)'
        assert needle in text
        text=text.replace(needle,'            results.append(result)\n            if clip_callback is not None:\n                clip_callback(result, split_event)\n            write_results(results, output_dir)')
    (DEST/name).write_text(text)
    manifest['files'][name]={'sha256':hashlib.sha256(source.encode()).hexdigest()}
(DEST/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
print('Copied GHR-VLM streaming and payment modules with a shared-clip callback.')

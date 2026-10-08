"""Build the frontend; optionally attach website assets from a private checkout."""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sanitize_html(html):
    for name in ('transitbox-data', 'transitbox-reid-meta'):
        pattern = r'(<script id="' + name + r'"[^>]*>).*?(</script>)'
        html, count = re.subn(pattern, lambda m: m[1] + '{}' + m[2], html, flags=re.S)
        if count != 1:
            raise ValueError(f'Expected one {name} tag')
    html = re.sub(r'(<span id="reid-run-summary">).*?(</span>)',
                  lambda m: m[1] + 'No local data loaded.' + m[2], html, flags=re.S)
    html = html.replace('C3_1 and C3_3 are the original videos.',
                        'Video and result files are supplied locally.')
    html = html.replace('Local playback workspace. No website has been published.',
                        'Videos and passenger records are not included in the public site.')
    return html


def check_html(html):
    for name in ('transitbox-data', 'transitbox-reid-meta'):
        match = re.search(r'<script id="' + name + r'"[^>]*>(.*?)</script>', html, re.S)
        if not match or json.loads(match[1]) != {}:
            raise ValueError(f'Public HTML contains private data in {name}')
    if re.search(r'data:(?:image|video|audio)/[^"\s]*;base64,', html):
        raise ValueError('Public HTML contains embedded binary media')
    markup = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.S | re.I)
    if re.search(r'<(?:video|img|source)\b[^>]*\b(?:src|poster)\s*=', markup, re.I):
        raise ValueError('Public HTML contains a media source')


def build(output=None, private_assets=None):
    output = Path(output or ROOT/'.publish/public-site').resolve()
    # Only this generated directory can be replaced; private public/ exports stay local.
    if output != (ROOT/'.publish/public-site').resolve():
        raise ValueError('Use .publish/public-site for the public frontend')
    html = sanitize_html((ROOT/'index.html').read_text())
    check_html(html)
    if private_assets is not None:
        assets = Path(private_assets).resolve()
        from tools.build_site import read_tag, write_tag
        from tools.check_publication import check
        from tools.check_redaction import check as check_redaction
        # Verify existing video/result assets without running inference or encoding.
        check_redaction(assets)
        check(assets)
        snapshot = (assets/'index.html').read_text()
        for name in ('transitbox-data', 'transitbox-reid-meta'):
            html = write_tag(html, name, read_tag(snapshot, name))
        manifest = json.loads((assets/'runtime/reid_results.json').read_text())
        summary = f"{manifest['completed']} clips processed with TransReID."
        html = re.sub(r'(<span id="reid-run-summary">).*?(</span>)',
                      lambda m: m[1] + summary + m[2], html, flags=re.S)
        html = html.replace('Video and result files are supplied locally.',
                            'Source videos, passenger clips and saved results are synchronized to video time.')
        html = html.replace('Videos and passenger records are not included in the public site.',
                            'Detection, identity and payment results were computed before publication.')
        html = html.replace('One passenger track list shared by ReID and payment detection.',
                            'One passenger track list shared by ReID and payment detection. Faces are blurred.')
        library = '<script src="https://cdn.jsdelivr.net/npm/hls.js@1.6.13/dist/hls.min.js" crossorigin="anonymous"></script>\n'
        if library not in html:
            html = html.replace('<script>\n', library + '<script>\n', 1)
    if output.exists():
        shutil.rmtree(output)
    if private_assets is not None:
        shutil.copytree(assets, output, ignore=shutil.ignore_patterns(
            '.git', '.build-cache.json', 'encoding.json', 'progress.txt'))
    else:
        output.mkdir(parents=True)
    (output/'index.html').write_text(html)
    (output/'.nojekyll').write_text('')
    if private_assets is not None:
        check_redaction(output)
        report = check(output)
        (output/'publication.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f"Built website: {report['clips']} clips, {report['size_mib']} MiB; assets remain outside source Git history")
    else:
        print(f'Built public frontend: {output} (no datasets or media)')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-assets', type=Path)
    args = parser.parse_args()
    build(private_assets=args.private_assets)

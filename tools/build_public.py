"""Build a public frontend without local videos, images or result records."""
from pathlib import Path
import argparse
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


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


def build(output=None):
    output = Path(output or ROOT/'.publish/public-site').resolve()
    # Only this generated directory can be replaced; private public/ exports stay local.
    if output != (ROOT/'.publish/public-site').resolve():
        raise ValueError('Use .publish/public-site for the public frontend')
    html = sanitize_html((ROOT/'index.html').read_text())
    check_html(html)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output/'index.html').write_text(html)
    (output/'.nojekyll').write_text('')
    print(f'Built public frontend: {output} (no datasets or media)')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()

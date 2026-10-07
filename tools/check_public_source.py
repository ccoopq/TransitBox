"""Reject data assets and embedded records in public source and Git history."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_public import check_html

SUFFIXES = {'.py', '.js', '.cjs', '.html', '.css', '.md', '.txt', '.yml', '.yaml', '.toml', '.sh'}
SPECIAL = {'.gitignore', '.gitattributes', 'LICENSE'}
PRIVATE_DIRS = {'public', 'runtime', 'media', 'assets', 'previews', 'models', 'dataset', 'data', '.publish', 'node_modules', '__pycache__'}


def check_file(name, content):
    path = Path(name)
    if any(part in PRIVATE_DIRS or part.startswith('outputs_') for part in path.parts):
        raise ValueError(f'Private/generated path in public source: {name}')
    json_allowed = name in ('package.json', 'package-lock.json') or (
        name.startswith('vendor/') and path.name == 'PROVENANCE.json')
    if path.suffix not in SUFFIXES and path.name not in SPECIAL and not json_allowed:
        raise ValueError(f'Non-source file in public repository: {name}')
    if b'\0' in content or len(content) > 2_000_000:
        raise ValueError(f'Binary or oversized content: {name}')
    if path.suffix == '.html':
        check_html(content.decode())


def check(root=ROOT, history=False):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    if (root/'.git').exists():
        names = git('ls-files', '-z').decode().split('\0')
    else:
        names = [str(p.relative_to(root)) for p in root.rglob('*')
                 if p.is_file() and not any(part in PRIVATE_DIRS for part in p.relative_to(root).parts)]
    for name in filter(None, names):
        path = root/name
        if path.is_symlink():
            raise ValueError(f'Symlink in public source: {name}')
        check_file(name, path.read_bytes())
    if history:
        seen = set()
        for commit in git('rev-list', '--all').decode().splitlines():
            for entry in git('ls-tree', '-rz', commit).split(b'\0'):
                if not entry:
                    continue
                header, name = entry.split(b'\t', 1)
                mode, kind, blob = header.split()
                if mode != b'100644' and mode != b'100755':
                    raise ValueError(f'Non-regular file in public history: {name.decode()}')
                key = (name, blob)
                if key not in seen:
                    check_file(name.decode(), git('cat-file', 'blob', blob.decode()))
                    seen.add(key)
    print('PASS: public source contains only code; no media or embedded records' +
          (' (all Git history checked)' if history else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=ROOT)
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    check(args.root, args.history)

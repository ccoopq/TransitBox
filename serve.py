"""Local preview server with byte-range support for the original MP4 videos."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT

class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVER_ROOT), **kwargs)

    def send_head(self):
        path = Path(self.translate_path(self.path))
        self._byte_range = None
        range_header = self.headers.get('Range')
        if not range_header or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
        if not match or size == 0 or not any(match.groups()):
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = min(int(end_text) if end_text else size - 1, size - 1)
        else:
            start = max(0, size - int(end_text))
            end = size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        file = path.open('rb')
        file.seek(start)
        self._byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Last-Modified', self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return file

    def copyfile(self, source, output):
        if self._byte_range is None:
            return super().copyfile(source, output)
        remaining = self._byte_range[1] - self._byte_range[0] + 1
        while remaining > 0:
            chunk = source.read(min(remaining, 256 * 1024))
            if not chunk:
                break
            output.write(chunk)
            remaining -= len(chunk)

    def do_GET(self):
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel range requests when seeking or switching videos.

    def log_message(self, format, *args):
        if len(args) > 1 and str(args[1]).startswith(('4', '5')):
            super().log_message(format, *args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--directory',type=Path,default=None)
    args = parser.parse_args()
    SERVER_ROOT = args.directory.resolve() if args.directory else ROOT if (ROOT/'media/C3_1.mp4').is_file() else ROOT/'public'
    try:
        server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    except OSError as error:
        parser.exit(1,f'TransitBox server could not start: {error}\n')
    print(f'TransitBox: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

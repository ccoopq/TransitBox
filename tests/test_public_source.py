import unittest

from tools.build_public import check_html, sanitize_html
from tools.check_public_source import check_file


class PublicSourceTests(unittest.TestCase):
    def test_public_html_removes_both_datasets_and_model_metadata(self):
        html = '<script id="transitbox-data">{"camera":{"clips":[1]}}</script>'
        html += '<script id="transitbox-reid-meta">{"checkpoint":"private"}</script>'
        html += '<span id="reid-run-summary">Private run results</span>'
        clean = sanitize_html(html)
        check_html(clean)
        self.assertNotIn('Private run results', clean)
        self.assertNotIn('checkpoint', clean)

    def test_rejects_data_even_when_disguised_as_source(self):
        for name in ('public/index.html', 'runtime/reid_results.json',
                     'video.mp4', 'image.jpg', 'data.py'):
            content = b'\0private bytes' if name == 'data.py' else b'private data'
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_file(name, content)

    def test_rejects_inline_records_and_media(self):
        clean = '<script id="transitbox-data">{}</script>'
        clean += '<script id="transitbox-reid-meta">{}</script>'
        for html in (clean.replace('{}', '{"camera":1}', 1),
                     clean + '<video src="private.mp4"></video>',
                     clean + '<img src="data:image/jpeg;base64,YQ==">'):
            with self.subTest(html=html), self.assertRaises(ValueError):
                check_html(html)


if __name__ == '__main__':
    unittest.main()

"""Regression tests for checks that must fail rather than report false success."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = load_module("quality_api", "docs/check_api.py")
site = load_module("quality_site", "docs/check_site.py")
decode = load_module("quality_decode", "tests/decode_regression.py")
smoke = load_module("quality_smoke", "tests/smoke.py")


class DecodeCheckTests(unittest.TestCase):
    def test_empty_fixture_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(decode.main([], directory), 1)

    def test_missing_fixture_fails_before_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in decode.EXPECTED:
                if name != "tone.mp3":
                    (Path(directory) / name).touch()
            with mock.patch.object(decode, "digest") as digest, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(decode.main([], directory), 1)
            digest.assert_not_called()
            self.assertIn("tone.mp3", output.getvalue())

    def test_update_mode_does_not_hide_missing_fixtures(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(decode.main(["--update"], directory), 1)


class ApiCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "src").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "sound_rs.pyi").write_text('''__all__ = ["play", "Player"]
def play(): ...
class Player:
    def __init__(self): ...
    def stop(self): ...
    @property
    def volume(self): ...
    @volume.setter
    def volume(self, value): ...
''', encoding="utf-8")
        (self.root / "src/lib.rs").write_text('''#[pymodule]
fn sound_rs() {
    m.add_class::<Player>()?;
    m.add_function(wrap_pyfunction!(play, &m)?)?;
}
#[pymethods]
impl Player {
    #[new]
    fn new() {}
    fn stop() {}
    #[getter]
    fn volume() {}
    #[setter]
    fn set_volume() {}
}
''', encoding="utf-8")
        self.page = self.root / "docs/api.html"
        self.page.write_text('<main><h2 id="play">play</h2><h2 id="player">Player</h2>stop() volume</main>', encoding="utf-8")

    def test_matching_source_stub_and_docs_pass(self):
        self.assertEqual(api.api_issues(self.root), [])

    def test_registered_function_missing_from_stub_fails(self):
        path = self.root / "src/lib.rs"
        path.write_text(path.read_text() + '\nm.add_function(wrap_pyfunction!(new_api, &m)?)?;', encoding="utf-8")
        self.assertIn("Rust API missing from stub: new_api", api.api_issues(self.root))

    def test_unregistered_stub_function_fails(self):
        path = self.root / "src/lib.rs"
        path.write_text(path.read_text().replace('m.add_function(wrap_pyfunction!(play, &m)?)?;', ''), encoding="utf-8")
        self.assertIn("Stub API not registered in Rust: play", api.api_issues(self.root))

    def test_method_missing_from_stub_fails(self):
        path = self.root / "src/lib.rs"
        path.write_text(path.read_text().replace('    fn stop() {}', '    fn stop() {}\n    fn seek() {}'), encoding="utf-8")
        self.assertIn("Rust member missing from stub: Player.seek", api.api_issues(self.root))

    def test_context_method_missing_from_stub_fails(self):
        path = self.root / "src/lib.rs"
        path.write_text(path.read_text().replace('    fn stop() {}', '    fn stop() {}\n    fn __exit__() {}'), encoding="utf-8")
        self.assertIn("Rust member missing from stub: Player.__exit__", api.api_issues(self.root))

    def test_member_documented_in_wrong_section_fails(self):
        self.page.write_text('<main><h2 id="play">play stop()</h2><h2 id="player">Player</h2>volume</main>', encoding="utf-8")
        self.assertIn("Undocumented member: Player.stop", api.api_issues(self.root))

    def test_navigation_mention_is_not_api_documentation(self):
        self.page.write_text('<nav>play</nav><main><h2 id="player">Player</h2>stop() volume</main>', encoding="utf-8")
        self.assertIn("Undocumented API: play", api.api_issues(self.root))


class SiteCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "index.html").write_text('''<html><head>
<link rel="canonical" href="https://neksand.github.io/sound_rs/index.html">
<link rel="alternate" type="text/markdown" href="index.html.md">
<link rel="describedby" href="llms.txt">
</head><body><h1 id="main">Hello</h1><a href="#main">Top</a></body></html>''', encoding="utf-8")
        (self.root / "index.html.md").write_text("# Hello\n", encoding="utf-8")
        (self.root / "llms.txt").write_text("[Overview](https://neksand.github.io/sound_rs/index.html.md)\n", encoding="utf-8")
        (self.root / "search.json").write_text(json.dumps([{"url": "index.html", "title": "Hello", "body": "Hello", "section": "Start"}]), encoding="utf-8")
        (self.root / "sitemap.xml").write_text('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://neksand.github.io/sound_rs/index.html</loc></url></urlset>', encoding="utf-8")

    def edit_html(self, suffix):
        path = self.root / "index.html"
        path.write_text(path.read_text() + suffix, encoding="utf-8")

    def test_valid_site_passes(self):
        self.assertEqual(site.site_issues(self.root), [])

    def test_broken_local_link_fails(self):
        self.edit_html('<a href="missing.html">Missing</a>')
        self.assertTrue(any("missing local target" in item for item in site.site_issues(self.root)))

    def test_broken_anchor_fails(self):
        self.edit_html('<a href="#absent">Missing</a>')
        self.assertTrue(any("missing anchor" in item for item in site.site_issues(self.root)))

    def test_encoded_anchor_is_resolved(self):
        self.edit_html('<a href="#m%61in">Encoded</a>')
        self.assertEqual(site.site_issues(self.root), [])

    def test_duplicate_id_fails(self):
        self.edit_html('<div id="main">Duplicate</div>')
        self.assertTrue(any("duplicate id" in item for item in site.site_issues(self.root)))

    def test_external_url_is_not_fetched(self):
        self.edit_html('<a href="https://example.invalid/unreachable">External</a>')
        self.assertEqual(site.site_issues(self.root), [])

    def test_missing_markdown_counterpart_fails(self):
        (self.root / "index.html.md").unlink()
        self.assertTrue(any("Markdown counterpart" in item for item in site.site_issues(self.root)))

    def test_search_must_cover_every_page(self):
        (self.root / "search.json").write_text("[]", encoding="utf-8")
        self.assertTrue(any("search.json: page coverage" in item for item in site.site_issues(self.root)))

    def test_sitemap_must_cover_every_page(self):
        (self.root / "sitemap.xml").write_text("<urlset />", encoding="utf-8")
        self.assertTrue(any("sitemap.xml: page coverage" in item for item in site.site_issues(self.root)))

    def test_llms_must_cover_every_markdown_page(self):
        (self.root / "llms.txt").write_text("# Empty", encoding="utf-8")
        self.assertTrue(any("llms.txt: missing pages" in item for item in site.site_issues(self.root)))


class HeadlessSmokeTests(unittest.TestCase):
    def test_detected_microphone_does_not_open_a_stream(self):
        import sound_rs
        def forbidden_device(*args, **kwargs):
            raise AssertionError("sound device opened")

        # Preserve method descriptors for signature-only checks added by the
        # recorder workstream while still rejecting constructor calls.
        forbidden_device.record_to = sound_rs.Recorder.record_to
        with mock.patch.dict(os.environ, {"SOUND_RS_PLAY": "0"}), \
                mock.patch.object(sound_rs, "list_devices", return_value=[]), \
                mock.patch.object(sound_rs, "list_input_devices", return_value=[(0, "simulated", True)]), \
                mock.patch.object(sound_rs, "Recorder", new=forbidden_device), \
                mock.patch.object(sound_rs, "Player", new=forbidden_device), \
                contextlib.redirect_stdout(io.StringIO()):
            smoke.main()


if __name__ == "__main__":
    unittest.main()

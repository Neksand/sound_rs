"""CLI regression: JSON info and WAV conversion through the console entry point."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import sound_rs

ROOT = pathlib.Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
CLI = "import sound_rs, sys; sys.exit(sound_rs.cli())"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-c", CLI, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
    )


class CliTests(unittest.TestCase):
    def test_info_json(self):
        result = run_cli("info", FIXTURES / "tone.wav", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(
            set(data),
            {
                "file",
                "sample_rate",
                "channels",
                "duration_seconds",
                "frames",
                "codec",
                "bit_depth",
            },
        )
        self.assertEqual((data["sample_rate"], data["channels"]), (44100, 2))
        self.assertEqual((data["codec"], data["bit_depth"]), ("pcm", 16))
        self.assertGreater(data["duration_seconds"], 0)

    def test_convert_writes_readable_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "out.wav"
            result = run_cli(
                "convert", FIXTURES / "tone.mp3", output, "--subtype", "pcm24"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            audio, rate = sound_rs.load(output)
            self.assertEqual((rate, audio.shape[1]), (44100, 2))
            self.assertGreater(len(audio), 0)

    def test_bad_usage_exits_two(self):
        result = run_cli("convert", FIXTURES / "tone.wav")
        self.assertEqual(result.returncode, 2)
        self.assertIn("convert needs", result.stderr)

        result = run_cli(
            "convert", FIXTURES / "tone.wav", "out.wav", "--subtype", "mp3"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown subtype", result.stderr)


if __name__ == "__main__":
    unittest.main()

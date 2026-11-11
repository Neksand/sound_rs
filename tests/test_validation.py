"""Hardware-independent Python boundary regressions: python tests/test_validation.py."""

import pathlib
import struct
import tempfile
import threading
import sys
import unittest

import numpy as np
import sound_rs


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = pathlib.Path(self.directory.name) / "output.wav"
        self.fixture = pathlib.Path(__file__).parent / "fixtures" / "tone.wav"

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(sound_rs.SoundRsError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def test_invalid_write_preserves_existing_file(self):
        sentinel = b"existing file must survive rejected input"
        cases = [
            (np.zeros((1, 0), dtype=np.float32), 44100),
            (np.zeros((1, 65536), dtype=np.float32), 44100),
            (np.zeros(1, dtype=np.float32), 0),
            (np.zeros((1, 2), dtype=np.float32), 2**32 - 1),
            (np.array([np.nan], dtype=np.float32), 44100),
            (np.array([np.inf], dtype=np.float32), 44100),
            (np.array([-np.inf], dtype=np.float32), 44100),
        ]
        for samples, rate in cases:
            with self.subTest(shape=samples.shape, rate=rate, samples=samples[:1]):
                self.output.write_bytes(sentinel)
                self.assert_code("usage", sound_rs.write, str(self.output), samples, rate)
                self.assertEqual(self.output.read_bytes(), sentinel)

    def test_write_io_error(self):
        path = self.output.parent / "missing" / "output.wav"
        self.assert_code("io", sound_rs.write, str(path), np.zeros(1, dtype=np.float32), 44100)

    def test_pcm24_padding_is_in_riff_size(self):
        sound_rs.write(str(self.output), np.array([0.5], dtype=np.float32), 44100, "pcm24")
        data = self.output.read_bytes()
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], len(data) - 8)
        self.assertEqual(struct.unpack_from("<I", data, 40)[0], 3)

    def test_strided_readonly_array_roundtrip(self):
        original = np.linspace(-0.9, 0.9, 36, dtype=np.float32).reshape(6, 6)[::-2, ::2]
        original.flags.writeable = False
        sound_rs.write(str(self.output), original, 44100, "float32")
        actual, rate = sound_rs.load(str(self.output))
        self.assertEqual(rate, 44100)
        np.testing.assert_array_equal(actual, original)

    def test_dtype_remains_strict(self):
        with self.assertRaises(TypeError):
            sound_rs.write(str(self.output), np.zeros(2, dtype=np.float64), 44100)

    def test_write_releases_gil(self):
        start = threading.Event()
        ready = threading.Event()
        progressed = threading.Event()

        def worker():
            ready.set()
            start.wait()
            progressed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        interval = sys.getswitchinterval()
        try:
            self.assertTrue(ready.wait(2))
            samples = np.zeros(2_000_000, dtype=np.float32)
            # Prevent a regular Python bytecode switch from satisfying the test.
            sys.setswitchinterval(10)
            start.set()
            sound_rs.write(str(self.output), samples, 44100)
            ran_during_write = progressed.is_set()
        finally:
            sys.setswitchinterval(interval)
            start.set()
            thread.join(2)
        self.assertTrue(ran_during_write, "WAV encoding/write must release the GIL")

    def test_invalid_array_layout_is_rejected_before_device_access(self):
        self.assert_code("usage", sound_rs.play_array, np.zeros((0, 1), dtype=np.float32), 44100)
        self.assert_code("usage", sound_rs.play_array, np.zeros((65536, 1), dtype=np.float32), 44100)
        self.assert_code("usage", sound_rs.play_array, np.zeros(1, dtype=np.float32), 0)

    def test_load_options_rejected_before_file_access(self):
        for key in ("offset", "duration"):
            for value in (-1.0, float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value):
                    self.assert_code("usage", sound_rs.load, "missing.wav", **{key: value})
        for paths in ([], [str(self.fixture)]):
            self.assert_code("usage", sound_rs.load_many, paths, sample_rate=0)

    def test_zero_length_window_and_batch_remain_valid(self):
        data, rate = sound_rs.load(str(self.fixture), duration=0)
        self.assertEqual((data.shape, rate), ((0, 2), 44100))
        self.assertEqual(sound_rs.load_many([]), [])

    def test_invalid_synthesis_does_not_open_a_device(self):
        for frequency, duration, rate in [
            (float("nan"), 0.1, 44100),
            (float("inf"), 0.1, 44100),
            (-1.0, 0.1, 44100),
            (440.0, -1.0, 44100),
            (440.0, float("nan"), 44100),
            (440.0, float("inf"), 44100),
            (440.0, 0.1, 0),
        ]:
            with self.subTest(frequency=frequency, duration=duration, rate=rate):
                self.assert_code("usage", sound_rs.play_sine_wave, frequency, duration, rate)


if __name__ == "__main__":
    unittest.main()

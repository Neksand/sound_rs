"""Container/format coverage: AIFF, CAF and ALAC.

Generated from `tone.wav`, so the lossless formats must decode to the same
samples through both the file and bytes paths.
"""

import pathlib
import unittest

import numpy as np

import sound_rs

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
REFERENCE = FIXTURES / "tone.wav"

# Everything here is bit-exact with the reference WAV.
FORMATS = ("tone.aiff", "tone_alac.m4a", "tone.caf")


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REFERENCE.is_file():
            raise AssertionError(f"missing reference {REFERENCE.name}")
        cls.reference, _ = sound_rs.load(REFERENCE)
        cls.duration = sound_rs.get_audio_info(REFERENCE)[2]

    def test_format_fixtures(self):
        for name in FORMATS:
            path = FIXTURES / name
            with self.subTest(fixture=name):
                self.assertTrue(path.is_file(), f"{name} fixture is missing")

                self.assertTrue(sound_rs.is_supported_format(path))
                rate, channels, duration = sound_rs.get_audio_info(path)
                self.assertEqual((rate, channels), (44100, 2))
                self.assertAlmostEqual(duration, self.duration, delta=0.02)

                audio, decoded_rate = sound_rs.load(path)
                self.assertEqual(decoded_rate, 44100)
                self.assertEqual(audio.dtype, np.float32)
                self.assertEqual(audio.shape, self.reference.shape)
                self.assertTrue(np.array_equal(audio, self.reference))

                from_bytes, bytes_rate = sound_rs.load_bytes(path.read_bytes())
                self.assertEqual(bytes_rate, decoded_rate)
                self.assertTrue(np.array_equal(from_bytes, audio))


if __name__ == "__main__":
    unittest.main()

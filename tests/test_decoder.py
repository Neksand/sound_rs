"""Incremental decoder coverage: blocks, resampling, mono and EOF."""

import pathlib
import unittest

import numpy as np

import sound_rs

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class DecoderTests(unittest.TestCase):
    def test_blocks_match_full_decode(self):
        for name in ("tone.wav", "tone.flac", "tone.m4a", "tone.mp3"):
            path = FIXTURES / name
            with self.subTest(fixture=name):
                full, rate = sound_rs.load(path)
                with sound_rs.Decoder(path, block_frames=1000) as decoder:
                    self.assertEqual(decoder.sample_rate, rate)
                    self.assertEqual(decoder.channels, full.shape[1])
                    self.assertGreater(decoder.duration or 0, 0)
                    blocks = list(decoder)
                self.assertTrue(blocks)
                self.assertTrue(all(block.shape[1] == full.shape[1] for block in blocks))
                self.assertTrue(all(block.dtype == np.float32 for block in blocks))
                decoded = np.concatenate(blocks)
                # AAC-in-MP4 can retain tail padding in the whole-file path;
                # the streaming path trims it. Compare the common prefix.
                common = min(len(decoded), len(full))
                self.assertTrue(np.array_equal(decoded[:common], full[:common]))
                if name != "tone.m4a":
                    self.assertEqual(decoded.shape, full.shape)

    def test_read_returns_requested_sizes_and_empty_eof(self):
        path = FIXTURES / "tone.flac"
        with sound_rs.Decoder(path, block_frames=333) as decoder:
            self.assertEqual(decoder.read(100).shape, (100, 2))
            first = decoder.read(1000)
            self.assertEqual(first.shape, (1000, 2))
            rest = list(decoder)
            self.assertEqual(sum(len(block) for block in rest), 11025 - 1100)
            self.assertEqual(decoder.read(10).shape, (0, 2))
            self.assertEqual(decoder.read(0).shape, (0, 2))

    def test_resampling_and_mono_match_full_decode(self):
        path = FIXTURES / "tone.wav"
        expected, expected_rate = sound_rs.load(path, sample_rate=16000, mono=True)
        with sound_rs.Decoder(path, block_frames=700, sample_rate=16000, mono=True) as decoder:
            self.assertEqual(decoder.sample_rate, 16000)
            self.assertEqual(decoder.channels, 1)
            actual = np.concatenate(list(decoder))
        self.assertEqual(actual.shape, expected.shape)
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), 1e-3)

    def test_closed_decoder_rejects_read(self):
        decoder = sound_rs.Decoder(FIXTURES / "tone.wav")
        decoder.close()
        with self.assertRaises(sound_rs.SoundRsError) as raised:
            decoder.read()
        self.assertEqual(raised.exception.code, "usage")

    def test_invalid_options(self):
        path = FIXTURES / "tone.wav"
        for kwargs in (
            {"block_frames": 0},
            {"sample_rate": 0},
        ):
            with self.subTest(**kwargs), self.assertRaises(sound_rs.SoundRsError) as raised:
                sound_rs.Decoder(path, **kwargs)
            self.assertEqual(raised.exception.code, "usage")

    def test_channel_layout_is_known_before_first_read(self):
        # AAC-in-MP4 has no channel layout in the container; the decoder primes
        # one packet so the public property is not zero.
        with sound_rs.Decoder(FIXTURES / "tone.m4a") as decoder:
            self.assertEqual(decoder.channels, 2)


if __name__ == "__main__":
    unittest.main()

"""Opus and WebM coverage: Ogg, Matroska/WebM, multichannel and streaming."""

import pathlib
import unittest

import numpy as np

import sound_rs

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

OGG = ("tone_opus_mono.opus", "tone_opus_stereo.opus", "speech_opus_mono.opus")
WEBM = ("tone_opus_mono.webm", "tone_opus_stereo.webm")


class OpusTests(unittest.TestCase):
    def test_ogg_opus(self):
        for name in OGG:
            path = FIXTURES / name
            with self.subTest(fixture=name):
                self.assertTrue(sound_rs.is_supported_format(path))
                info = sound_rs.probe(path)
                self.assertEqual(info["codec"], "opus")
                self.assertEqual(info["sample_rate"], 48000)
                self.assertIn(info["channels"], (1, 2))
                self.assertAlmostEqual(info["duration_seconds"], 3.0, delta=0.01)

                audio, rate = sound_rs.load(path)
                self.assertEqual(rate, 48000)
                self.assertEqual(audio.shape, (144000, info["channels"]))
                from_bytes, byte_rate = sound_rs.load_bytes(path.read_bytes())
                self.assertEqual(byte_rate, rate)
                np.testing.assert_array_equal(from_bytes, audio)

    def test_webm_opus(self):
        for name in WEBM:
            path = FIXTURES / name
            with self.subTest(fixture=name):
                info = sound_rs.probe(path)
                self.assertEqual(info["codec"], "opus")
                self.assertEqual(info["sample_rate"], 48000)
                audio, rate = sound_rs.load(path)
                self.assertEqual(rate, 48000)
                self.assertEqual(audio.shape[0], 144648)  # one Opus packet tail
                from_bytes, _ = sound_rs.load_bytes(path.read_bytes())
                np.testing.assert_array_equal(from_bytes, audio)

    def test_ogg_and_webm_share_the_common_prefix(self):
        for mono, webm in zip(
            ("tone_opus_mono.opus", "tone_opus_stereo.opus"),
            WEBM,
        ):
            ogg_audio, _ = sound_rs.load(FIXTURES / mono)
            webm_audio, _ = sound_rs.load(FIXTURES / webm)
            prefix = min(len(ogg_audio), len(webm_audio))
            np.testing.assert_array_equal(ogg_audio, webm_audio[:prefix])

    def test_opus_multichannel(self):
        audio, rate = sound_rs.load(FIXTURES / "tone_opus_51.opus")
        self.assertEqual((rate, audio.shape), (48000, (144000, 6)))

    def test_decoder_matches_whole_file(self):
        path = FIXTURES / "tone_opus_stereo.opus"
        full, _ = sound_rs.load(path)
        with sound_rs.Decoder(path, block_frames=1000) as decoder:
            self.assertEqual(decoder.channels, 2)
            blocks = list(decoder)
        np.testing.assert_array_equal(np.concatenate(blocks), full)

    def test_batch_skip(self):
        clips = sound_rs.load_many(
            [FIXTURES / "tone_opus_mono.opus", FIXTURES / "missing.opus"],
            on_error="skip",
        )
        self.assertEqual(clips[0][0].shape, (144000, 1))
        self.assertIsNone(clips[1])


if __name__ == "__main__":
    unittest.main()

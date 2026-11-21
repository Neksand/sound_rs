#!/usr/bin/env python3
"""Decode regression: fixed fixtures must decode to exactly the same PCM.

Any change in a decoder, the gapless trimming or the channel handling moves one
of these hashes, which is the point. Re-run with `--update` only when the new
output is verified against ffmpeg by hand.

    python3 tests/decode_regression.py
    python3 tests/decode_regression.py --update
"""

import hashlib
import os
import sys

import numpy as np

import sound_rs

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

# (frames, channels, rate, sha256 of the rounded float32 PCM)
EXPECTED = {
    "tone.wav": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "tone24.wav": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "six.wav": (11025, 6, 44100, '6700c2b6b00becfc8a7575d2c1111886ec8edd0bccf310cbfe7f945d85b9e69b'),
    "tone.flac": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "tone.aiff": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "tone.caf": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "tone_opus_mono.opus": (144000, 1, 48000, 'ae1e833631793b4a683783d23e2768c3baf3d9452046c997660815b9ab91df61'),
    "tone_opus_stereo.opus": (144000, 2, 48000, '9f458862625f8f187242def39eac2eee0a7eef9abc085e43a47a40d4e44ba4bb'),
    "tone_opus_mono.webm": (144648, 1, 48000, '878db9cacd212454cda22f948d4954b994eb97d4d648fe729a9699599649d535'),
    "tone_opus_stereo.webm": (144648, 2, 48000, 'e4575d8e25d7827d8a090df49e5d8b37b0e17f834934cb79ab69b90eb204cfdd'),
    "tone_opus_51.opus": (144000, 6, 48000, 'fed9e889ae05b8157aeab68a0ce4848225207ad613ee228d326a679107018182'),
    "speech_opus_mono.opus": (144000, 1, 48000, '7100d4159bcbe0e0c57b5efd856b36ffb3195a4a099ceada9c0fb6a7f68c4095'),
    "tone_alac.m4a": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "tone.mp3": (11025, 2, 44100, '8bae1432bbd7faa87e45f647fe32357d9a014d2020cb8882376e68376b644851'),
    "tone.m4a": (11264, 2, 44100, '571871df7735a715dd907c602e72e4d1398b35304d8a270c139d2cb3132a792c'),
}


def digest(path):
    audio, rate = sound_rs.load(path)
    # 1e-4 is ~3 LSB at 16-bit scale: tight enough to catch a decoder change,
    # loose enough for the integer SIMD paths to differ by a bit across CPUs.
    rounded = np.round(audio.astype(np.float64), 4).astype(np.float32)
    return rounded.shape[0], rounded.shape[1], rate, hashlib.sha256(rounded.tobytes()).hexdigest()


def main(argv=None, fixtures=FIXTURES):
    update = "--update" in (sys.argv[1:] if argv is None else argv)
    missing = [name for name in sorted(EXPECTED) if not os.path.isfile(os.path.join(fixtures, name))]
    if missing:
        print("Missing required decode fixtures: " + ", ".join(missing))
        return 1
    failures = []
    for name in sorted(EXPECTED):
        path = os.path.join(fixtures, name)
        frames, channels, rate, sha = digest(path)
        want_frames, want_channels, want_rate, want_sha = EXPECTED[name]
        matching = update or sha == want_sha
        ok = (frames, channels, rate) == (want_frames, want_channels, want_rate) and matching
        status = "ok" if ok else "MISMATCH"
        print(f"{name:<14} {frames:6d} frames  {channels} ch  {rate} Hz  {sha[:16]}…  {status}")
        if not ok:
            failures.append(name)

    if update:
        print("\nPaste these into EXPECTED:")
        for name in sorted(EXPECTED):
            print(f'    "{name}": {digest(os.path.join(fixtures, name))},')
        return 0

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

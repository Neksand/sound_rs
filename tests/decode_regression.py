#!/usr/bin/env python3
"""Decode regression: fixed fixtures must decode to exactly the same PCM.

Any change in a decoder, the gapless trimming or the channel handling moves one
of these hashes, which is the point. Re-run with `--update` only when the new
output is verified against ffmpeg by hand.

    python3 tests/decode_regression.py
    python3 tests/decode_regression.py --update
"""

import glob
import hashlib
import os
import sys

import numpy as np

import sound_rs

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

# (frames, channels, rate, sha256 of the rounded float32 PCM)
MP3_RMS = 0.233488

EXPECTED = {
    "tone.wav": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "tone24.wav": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    "six.wav": (11025, 6, 44100, '6700c2b6b00becfc8a7575d2c1111886ec8edd0bccf310cbfe7f945d85b9e69b'),
    "tone.flac": (11025, 2, 44100, '93c7836cdfc70fc3791483df33fa00f11827cdba6f4afc24babd52055ad7a681'),
    # None: MP3 comes from a C decoder whose SIMD path can move the last
    # bits per CPU. Length and level are checked instead of a hash.
    "tone.mp3": (11025, 2, 44100, None),
    "tone.m4a": (11264, 2, 44100, '571871df7735a715dd907c602e72e4d1398b35304d8a270c139d2cb3132a792c'),
}


def digest(path):
    audio, rate = sound_rs.load(path)
    # 1e-4 is ~3 LSB at 16-bit scale: tight enough to catch a decoder change,
    # loose enough for the integer SIMD paths to differ by a bit across CPUs.
    rounded = np.round(audio.astype(np.float64), 4).astype(np.float32)
    return rounded.shape[0], rounded.shape[1], rate, hashlib.sha256(rounded.tobytes()).hexdigest()


def main():
    update = "--update" in sys.argv
    failures = []
    for path in sorted(glob.glob(os.path.join(FIXTURES, "*"))):
        name = os.path.basename(path)
        if name not in EXPECTED:
            continue
        frames, channels, rate, sha = digest(path)
        want_frames, want_channels, want_rate, want_sha = EXPECTED[name]
        matching = update or sha == want_sha
        if want_sha is None and not update:
            audio, _ = sound_rs.load(path)
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            matching = abs(rms - MP3_RMS) < 1e-3
            sha = f"rms {rms:.4f}"
        ok = (frames, channels, rate) == (want_frames, want_channels, want_rate) and matching
        status = "ok" if ok else "MISMATCH"
        print(f"{name:<14} {frames:6d} frames  {channels} ch  {rate} Hz  {sha[:16]}…  {status}")
        if not ok:
            failures.append(name)

    if update:
        print("\nPaste these into EXPECTED:")
        for path in sorted(glob.glob(os.path.join(FIXTURES, "*"))):
            name = os.path.basename(path)
            if name in EXPECTED:
                print(f'    "{name}": {digest(path)},')
        return 0

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
EXPECTED = {
    "tone.wav": (11025, 2, 44100, '5312e9b1adb1a06f90f709367bbbdebb12e43dc954f17609abde1a747e17243b'),
    "tone24.wav": (11025, 2, 44100, '5312e9b1adb1a06f90f709367bbbdebb12e43dc954f17609abde1a747e17243b'),
    "six.wav": (11025, 6, 44100, '6822e6d130067ffd2cecf61645c3eb6fdcd7c3b1bf96a28379b2b88b6adde592'),
    "tone.flac": (11025, 2, 44100, '5312e9b1adb1a06f90f709367bbbdebb12e43dc954f17609abde1a747e17243b'),
    "tone.mp3": (11025, 2, 44100, '51d1ec126fa537288e5ce161551887a7907da1d983848e3fb9093f87c0b35ef1'),
    "tone.m4a": (11264, 2, 44100, '5efd9ad25f90cc8963def96abfa255bc10f8802bfbccdd851df3977056d5e70a'),
}


def digest(path):
    audio, rate = sound_rs.load(path)
    rounded = np.round(audio.astype(np.float64), 6).astype(np.float32)
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
        ok = (frames, channels, rate) == (want_frames, want_channels, want_rate) and (
            update or sha == want_sha
        )
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

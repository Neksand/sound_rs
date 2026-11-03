"""Honest benchmark.

Playback is dominated by the track's own duration, so the number that matters
is the overhead: wall clock minus audio duration. Decode is measured on its
own, where the work actually is.

    python examples/benchmark.py            # sound_rs only
    python examples/benchmark.py --compare  # + sounddevice / soundfile if installed
"""

import argparse
import math
import os
import statistics
import struct
import tempfile
import time
import wave

import numpy as np
import sound_rs

RATE = 44100
DURATION = 3.0
RUNS = 5


def write_wav(path, seconds, rate=RATE):
    body = bytearray()
    for i in range(int(seconds * rate)):
        body += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(body))


def stdlib_decode(path):
    """Same work sound_rs does: 16-bit PCM -> interleaved float32."""
    with wave.open(path) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def bench(label, func, runs=RUNS):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    best = min(times)
    print(f"  {label:<28} {best * 1000:8.1f} ms  (median {statistics.median(times) * 1000:.1f} ms)")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="measure this file (mp3/flac/m4a...) instead of a generated 3s wav")
    parser.add_argument("--compare", action="store_true", help="also measure sounddevice/soundfile")
    args = parser.parse_args()

    if args.file:
        path = args.file
        duration = sound_rs.get_audio_info(path)[2]
        play = lambda: sound_rs.play_audio(path)
        stdlib = None
    else:
        path = os.path.join(tempfile.mkdtemp(), "tone.wav")
        write_wav(path, DURATION)
        duration = DURATION
        data = np.sin(2 * np.pi * 440 * np.arange(int(DURATION * RATE)) / RATE).astype(np.float32)
        play = lambda: sound_rs.play_array(data, RATE)
        stdlib = stdlib_decode

    print(f"\nDecode {duration:.1f}s {os.path.basename(path)}")
    bench("sound_rs.get_audio_info", lambda: sound_rs.get_audio_info(path))
    if stdlib:
        bench("stdlib wave + numpy", lambda: stdlib(path))

    if args.compare:
        try:
            import soundfile
            bench("soundfile.read", lambda: soundfile.read(path, dtype="float32"))
        except ImportError:
            print("  soundfile not installed, skipped")

    print(f"\nPlayback of the same tone (overhead = wall clock - {duration:.1f}s)")
    play_best = bench("sound_rs play", play)
    print(f"  -> sound_rs overhead {play_best - duration:+.3f} s")

    if args.compare and not args.file:
        try:
            import sounddevice as sd

            sd_time = bench("sounddevice.play + wait", lambda: sd.play(data, RATE) or sd.wait())
            print(f"  -> sounddevice overhead {sd_time - duration:+.3f} s")
        except ImportError:
            print("  sounddevice not installed, skipped")

    print()


if __name__ == "__main__":
    main()

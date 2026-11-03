"""Honest benchmark.

Playback is dominated by the track's own duration, so the number that matters
is the overhead: wall clock minus audio duration. Decode is where the work is,
so it is measured per format against soundfile (if installed).

    python examples/benchmark.py                      # generated 3s tone, decode + playback
    python examples/benchmark.py --compare            # + sounddevice / soundfile
    python examples/benchmark.py --files a.mp3 b.flac # decode-only table for real files
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


def bench(func, runs=RUNS):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def decode_table(files, compare):
    print(f"\n{'format':<6} {'size':>9} {'sound_rs':>10} {'soundfile':>10}  winner")
    for path in files:
        label = os.path.splitext(path)[1].lstrip(".").lower()
        size = f"{os.path.getsize(path) / 1e6:.1f} MB"
        ours, _ = bench(lambda p=path: sound_rs.get_audio_info(p))

        theirs = None
        if compare:
            try:
                import soundfile

                theirs, _ = bench(lambda p=path: soundfile.read(p, dtype="float32"))
            except Exception:
                theirs = None

        ours_s = f"{ours * 1000:.1f} ms"
        if theirs is None:
            print(f"{label:<6} {size:>9} {ours_s:>10} {'unsupported':>10}  sound_rs only")
        else:
            theirs_s = f"{theirs * 1000:.1f} ms"
            winner = "sound_rs" if ours < theirs else "soundfile"
            print(f"{label:<6} {size:>9} {ours_s:>10} {theirs_s:>10}  {winner} ({max(ours, theirs) / min(ours, theirs):.1f}x)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", help="files to decode (skips playback)")
    parser.add_argument("--compare", action="store_true", help="also measure sounddevice/soundfile")
    args = parser.parse_args()

    if args.files:
        decode_table(args.files, compare=True)
        print()
        return

    path = os.path.join(tempfile.mkdtemp(), "tone.wav")
    write_wav(path, DURATION)
    data = np.sin(2 * np.pi * 440 * np.arange(int(DURATION * RATE)) / RATE).astype(np.float32)

    print(f"\nDecode {DURATION:.1f}s 44.1 kHz mono WAV")
    ours, _ = bench(lambda: sound_rs.get_audio_info(path))
    print(f"  sound_rs.get_audio_info   {ours * 1000:6.2f} ms")
    theirs, _ = bench(lambda: stdlib_decode(path))
    print(f"  stdlib wave + numpy       {theirs * 1000:6.2f} ms")
    if args.compare:
        try:
            import soundfile

            sf, _ = bench(lambda: soundfile.read(path, dtype="float32"))
            print(f"  soundfile.read            {sf * 1000:6.2f} ms")
        except ImportError:
            print("  soundfile not installed, skipped")

    print(f"\nPlayback of the same {DURATION:.1f}s tone (overhead = wall clock - {DURATION:.1f}s)")
    play, _ = bench(lambda: sound_rs.play_array(data, RATE))
    print(f"  sound_rs                  {play:6.3f} s  overhead {play - DURATION:+.3f} s")

    if args.compare:
        try:
            import sounddevice as sd

            theirs, _ = bench(lambda: sd.play(data, RATE) or sd.wait())
            print(f"  sounddevice.play + wait   {theirs:6.3f} s  overhead {theirs - DURATION:+.3f} s")
        except ImportError:
            print("  sounddevice not installed, skipped")

    print()


if __name__ == "__main__":
    main()

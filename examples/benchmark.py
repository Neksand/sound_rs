"""Honest benchmark.

Playback is dominated by the track's own duration, so the number that matters
is the overhead: wall clock minus audio duration. Decode is where the work is,
so it is measured per format and checked for correctness against libsndfile.

    python examples/benchmark.py                      # generated 3s tone, decode + playback
    python examples/benchmark.py --compare            # + sounddevice / soundfile
    python examples/benchmark.py --files a.mp3 b.flac # decode-only table for real files
"""

import argparse
import math
import os
import resource
import statistics
import struct
import tempfile
import time
import wave

import numpy as np
import sound_rs

RATE = 44100
DURATION = 3.0
RUNS = 7


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
    """Returns (min, median, p95) in seconds. Min is the best case, p95 the tail."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    times.sort()
    p95 = times[max(0, math.ceil(0.95 * len(times)) - 1)]
    return times[0], statistics.median(times), p95


def peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if os.uname().sysname == "Darwin" else usage / 1024


def pcm_check(path):
    """Decode with both libraries and compare samples. Speed without this is meaningless."""
    try:
        import soundfile

        reference, _ = soundfile.read(path, dtype="float32", always_2d=True)
    except Exception:
        return "n/a"
    ours, _ = sound_rs.load(path)
    if ours.shape != reference.shape:
        return f"shape {ours.shape} vs {reference.shape}"
    diff = float(np.abs(ours - reference).max())
    return f"{diff:.1e}" if diff else "identical"


def decode_table(files):
    for path in files:
        label = os.path.splitext(path)[1].lstrip(".").lower()
        best, median, p95 = bench(lambda p=path: sound_rs.get_audio_info(p))
        ours_best, ours_med, ours_p95 = bench(lambda p=path: sound_rs.load(p))

        theirs = None
        try:
            import soundfile

            theirs = bench(lambda p=path: soundfile.read(p, dtype="float32"))
        except Exception:
            theirs = None

        print(f"\n{label}  {os.path.getsize(path) / 1e6:.1f} MB  {path}")
        print(f"  metadata  sound_rs  median {median * 1e3:7.2f} ms   p95 {p95 * 1e3:7.2f} ms")
        print(f"  decode    sound_rs  median {ours_med * 1e3:7.2f} ms   p95 {ours_p95 * 1e3:7.2f} ms")
        if theirs:
            print(
                f"  decode    soundfile median {theirs[1] * 1e3:7.2f} ms   p95 {theirs[2] * 1e3:7.2f} ms"
                f"   ({(theirs[1] / ours_med):.2f}x of ours)"
            )
        else:
            print("  decode    soundfile unsupported")
        print(f"  PCM check vs soundfile: {pcm_check(path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", help="files to decode (skips playback)")
    parser.add_argument("--compare", action="store_true", help="also measure sounddevice/soundfile")
    args = parser.parse_args()

    if args.files:
        decode_table(args.files)
        print(f"\npeak RSS (whole run): {peak_rss_mb():.0f} MB")
        print()
        return

    path = os.path.join(tempfile.mkdtemp(), "tone.wav")
    write_wav(path, DURATION)
    data = np.sin(2 * np.pi * 440 * np.arange(int(DURATION * RATE)) / RATE).astype(np.float32)

    print(f"\nDecode {DURATION:.1f}s 44.1 kHz mono WAV")
    _, ours, ours_p95 = bench(lambda: sound_rs.get_audio_info(path))
    print(f"  sound_rs.get_audio_info   median {ours * 1e3:6.2f} ms   p95 {ours_p95 * 1e3:6.2f} ms")
    _, theirs, _ = bench(lambda: stdlib_decode(path))
    print(f"  stdlib wave + numpy       median {theirs * 1e3:6.2f} ms")
    print(f"  PCM check vs stdlib:      {pcm_check(path)}")
    if args.compare:
        try:
            import soundfile

            _, sf, _ = bench(lambda: soundfile.read(path, dtype="float32"))
            print(f"  soundfile.read            median {sf * 1e3:6.2f} ms")
        except ImportError:
            print("  soundfile not installed, skipped")

    print(f"\nPlayback of the same {DURATION:.1f}s tone (overhead = wall clock - {DURATION:.1f}s)")
    best, median, p95 = bench(lambda: sound_rs.play_array(data, RATE), runs=5)
    print(
        f"  sound_rs                  median {median:6.3f} s   p95 {p95:6.3f} s"
        f"   overhead {median - DURATION:+.3f} s"
    )

    if args.compare:
        try:
            import sounddevice as sd

            _, theirs, theirs_p95 = bench(lambda: sd.play(data, RATE) or sd.wait(), runs=5)
            print(
                f"  sounddevice.play + wait   median {theirs:6.3f} s   p95 {theirs_p95:6.3f} s"
                f"   overhead {theirs - DURATION:+.3f} s"
            )
        except ImportError:
            print("  sounddevice not installed, skipped")

    print(f"\npeak RSS (whole run): {peak_rss_mb():.0f} MB\n")


if __name__ == "__main__":
    main()

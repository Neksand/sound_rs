"""Benchmark harness: every measurement runs in its own process.

Why a subprocess per measurement: peak RSS is a process-wide high-water mark,
so timing and memory cannot be measured in the same process without polluting
each other. Each worker below prints one JSON line and exits.

    python bench/run.py                 # sound_rs only, full report
    python bench/run.py --compare       # + soundfile / sounddevice / pydub
    python bench/run.py --reps 9        # more repetitions
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
RATE = 44100
SIZES = {"short": 3.0, "long": 600.0}


# --------------------------------------------------------------------------
# corpus


def build_corpus():
    os.makedirs(CORPUS, exist_ok=True)
    wavs = {}
    for name, seconds in SIZES.items():
        path = os.path.join(CORPUS, f"{name}.wav")
        if not os.path.exists(path):
            import wave

            body = bytearray()
            for i in range(int(seconds * RATE)):
                value = int(12000 * __import__("math").sin(2 * 3.14159265 * 440 * i / RATE))
                body += value.to_bytes(2, "little", signed=True)
                body += value.to_bytes(2, "little", signed=True)
            with wave.open(path, "w") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(bytes(body))
        wavs[name] = path

    files = dict(wavs)
    if shutil.which("ffmpeg"):
        for name, wav in wavs.items():
            for ext, args in (
                ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
                ("flac", []),
                ("m4a", ["-c:a", "aac"]),
            ):
                out = os.path.join(CORPUS, f"{name}.{ext}")
                if not os.path.exists(out):
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, *args, out],
                        check=True,
                    )
                files[f"{name}.{ext}"] = out
    return files


# --------------------------------------------------------------------------
# workers: run in a child process, print one JSON line


def worker(kind, target, extra):
    t0 = time.perf_counter()
    if kind == "import":
        __import__(target)
        result = {"seconds": time.perf_counter() - t0}
    elif kind == "metadata":
        import sound_rs

        sound_rs.get_audio_info(target)
        result = {"seconds": time.perf_counter() - t0}
    elif kind == "metadata_sf":
        import soundfile

        soundfile.info(target)
        result = {"seconds": time.perf_counter() - t0}
    elif kind == "decode":
        import sound_rs

        sound_rs.load(target)
        result = {"seconds": time.perf_counter() - t0}
    elif kind == "decode_sf":
        import soundfile

        soundfile.read(target, dtype="float32")
        result = {"seconds": time.perf_counter() - t0}
    elif kind == "pcm":
        import numpy as np
        import sound_rs
        import soundfile

        reference, _ = soundfile.read(target, dtype="float32", always_2d=True)
        ours, _ = sound_rs.load(target)
        n = min(len(ours), len(reference))
        span = min(n, 200_000)
        direct = float(np.abs(ours[:n][:span] - reference[:n][:span]).max())

        # Lossy decoders disagree by a few samples of delay, so also report the
        # best alignment within +-2 ms and the diff at that alignment.
        best_lag, best_diff = 0, direct
        for lag in range(-96, 97):
            a = ours[max(0, lag):max(0, lag) + span, 0]
            b = reference[max(0, -lag):max(0, -lag) + span, 0]
            width = min(len(a), len(b))
            if width < 100:
                continue
            diff = float(np.abs(a[:width] - b[:width]).max())
            if diff < best_diff:
                best_lag, best_diff = lag, diff

        result = {
            "max_diff": direct,
            "lag": best_lag,
            "aligned_diff": best_diff,
            "frames": int(len(ours)),
            "reference_frames": int(len(reference)),
        }
    elif kind == "play_overhead":
        import numpy as np
        import sound_rs

        seconds = float(extra)
        samples = (
            np.sin(2 * np.pi * 440 * np.arange(int(seconds * RATE)) / RATE)
            .astype(np.float32)
        )
        start = time.perf_counter()
        sound_rs.play_array(samples, RATE)
        result = {"seconds": time.perf_counter() - start}
    elif kind == "play_overhead_player":
        import numpy as np
        import sound_rs

        seconds = float(extra)
        samples = (
            np.sin(2 * np.pi * 440 * np.arange(int(seconds * RATE)) / RATE)
            .astype(np.float32)
        )
        player = sound_rs.Player()
        warm = np.zeros(2400, dtype=np.float32)
        player.play_array(warm, RATE)
        player.wait()
        start = time.perf_counter()
        player.play_array(samples, RATE)
        player.wait()
        result = {"seconds": time.perf_counter() - start}
        player.stop()
    elif kind == "play_overhead_sd":
        import numpy as np
        import sounddevice as sd

        seconds = float(extra)
        samples = (
            np.sin(2 * np.pi * 440 * np.arange(int(seconds * RATE)) / RATE)
            .astype(np.float32)
        )
        start = time.perf_counter()
        sd.play(samples, RATE)
        sd.wait()
        result = {"seconds": time.perf_counter() - start}
    elif kind == "first_sound":
        import sound_rs

        player = sound_rs.Player()
        start = time.perf_counter()
        player.play_file(target)
        while player.position <= 0.0:
            time.sleep(0.001)
        result = {"seconds": time.perf_counter() - start}
        player.stop()
    elif kind == "first_sound_sf":
        import sounddevice as sd
        import soundfile

        start = time.perf_counter()
        data, rate = soundfile.read(target, dtype="float32")
        sd.play(data, rate)
        while True:
            stream_time = sd.get_stream().time
            if stream_time > 0:
                break
            time.sleep(0.001)
        result = {"seconds": time.perf_counter() - start}
        sd.stop()
    elif kind == "batch":
        import sound_rs

        paths = extra.split(",")
        start = time.perf_counter()
        sound_rs.load_many(paths)
        result = {"seconds": time.perf_counter() - start, "files": len(paths)}
    elif kind == "batch_sf":
        import soundfile

        paths = extra.split(",")
        start = time.perf_counter()
        for path in paths:
            soundfile.read(path, dtype="float32")
        result = {"seconds": time.perf_counter() - start, "files": len(paths)}
    else:
        raise SystemExit(f"unknown worker {kind}")

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
        result["peak_mb"] = peak
    except ImportError:
        result["peak_mb"] = None
    print(json.dumps(result))


def spawn(kind, target="", extra="", python=None):
    # `target` is always passed (possibly empty) so `extra` never shifts into
    # its slot.
    command = [python or sys.executable, os.path.abspath(__file__), "--worker", kind, target]
    if extra:
        command.append(extra)
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    line = out.stdout.strip().splitlines()[-1]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def measure(kind, target="", extra="", reps=5):
    runs = [spawn(kind, target, extra) for _ in range(reps)]
    runs = [r for r in runs if r]
    if not runs:
        return None
    seconds = sorted(r["seconds"] for r in runs)
    p95 = seconds[max(0, -(-95 * len(seconds) // 100) - 1)]
    peaks = [r["peak_mb"] for r in runs if r.get("peak_mb")]
    return {
        "median": statistics.median(seconds),
        "p95": p95,
        "peak_mb": max(peaks) if peaks else None,
        "raw": runs[-1],
    }


def fmt(ms):
    return f"{ms * 1000:.2f} ms" if ms is not None else "-"


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--compare", action="store_true", help="also measure competitors")
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    if args.worker:
        kind, *rest = args.worker
        target = rest[0] if rest else ""
        extra = rest[1] if len(rest) > 1 else ""
        worker(kind, target, extra)
        return

    import importlib.util

    def available(module):
        return importlib.util.find_spec(module) is not None

    files = build_corpus()
    print(
        f"\nsound_rs benchmark — {platform.processor() or platform.machine()}, "
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}"
    )
    print(f"reps={args.reps}, corpus: {CORPUS}\n")

    # 1. import
    print("import time (cold process)")
    for module in ["sound_rs", "soundfile", "sounddevice"]:
        if module != "sound_rs" and not (args.compare and available(module)):
            continue
        result = measure("import", module, reps=max(3, args.reps))
        if result:
            print(
                f"  {module:<12} {fmt(result['median']):>10}   peak {result['peak_mb']:.0f} MB"
            )

    # 2. metadata and decode
    print("\nmetadata / decode (median, p95, peak RSS)")
    for name, path in sorted(files.items()):
        size_mb = os.path.getsize(path) / 1e6
        meta = measure("metadata", path, reps=args.reps)
        dec = measure("decode", path, reps=args.reps)
        line = f"  {name:<12} {size_mb:6.1f} MB  metadata {fmt(meta and meta['median']):>9}  decode {fmt(dec and dec['median']):>9} (p95 {fmt(dec and dec['p95'])})  peak {dec and dec['peak_mb']:.0f} MB"

        if args.compare:
            sf_meta = measure("metadata_sf", path, reps=args.reps) if available("soundfile") else None
            sf_dec = measure("decode_sf", path, reps=args.reps) if available("soundfile") else None
            if sf_dec:
                line += f"  | soundfile meta {fmt(sf_meta and sf_meta['median'])}, decode {fmt(sf_dec['median'])}"
            else:
                line += "  | soundfile unsupported"
        print(line)

        if available("soundfile") and dec:
            pcm = spawn("pcm", path)
            if pcm:
                detail = (
                    "identical"
                    if pcm["aligned_diff"] == 0
                    else f"max diff {pcm['aligned_diff']:.1e}"
                )
                if pcm["lag"]:
                    detail += f", offset {pcm['lag']} samples"
                print(
                    f"  {'':<12} PCM vs soundfile: {detail} "
                    f"(frames {pcm['frames']} vs {pcm['reference_frames']})"
                )

    # 3. playback
    print("\nplayback of a 3 s tone")
    ours = measure("play_overhead", extra="3.0", reps=3)
    if ours:
        print(
            f"  sound_rs     {ours['median']:6.3f} s  overhead {ours['median'] - 3.0:+.3f} s"
            f"   p95 {ours['p95']:6.3f} s"
        )
    persistent = measure("play_overhead_player", extra="3.0", reps=3)
    if persistent:
        print(
            f"  sound_rs Player  {persistent['median']:6.3f} s  overhead {persistent['median'] - 3.0:+.3f} s"
            f"   p95 {persistent['p95']:6.3f} s"
        )
    if args.compare and available("sounddevice"):
        theirs = measure("play_overhead_sd", extra="3.0", reps=3)
        if theirs:
            print(
                f"  sounddevice  {theirs['median']:6.3f} s  overhead {theirs['median'] - 3.0:+.3f} s"
                f"   p95 {theirs['p95']:6.3f} s"
            )

    # 4. streaming: memory and time to first sound, on the long file
    long_path = files.get("long.flac") or files.get("long.wav")
    if long_path:
        print(f"\nstreaming {os.path.basename(long_path)} ({SIZES['long']:.0f} s)")
        fs = measure("first_sound", long_path, reps=3)
        if fs:
            print(
                f"  sound_rs Player   first sound {fmt(fs['median']):>9}   peak {fs['peak_mb']:.0f} MB"
            )
        if args.compare and available("sounddevice") and available("soundfile"):
            fs_sf = measure("first_sound_sf", long_path, reps=3)
            if fs_sf:
                print(
                    f"  soundfile+sd      first sound {fmt(fs_sf['median']):>9}   peak {fs_sf['peak_mb']:.0f} MB"
                )

    # 5. batch decode
    batch_paths = ",".join(path for name, path in sorted(files.items()) if name.endswith(".flac"))
    if batch_paths:
        print(f"\nbatch decode ({len(batch_paths.split(','))} FLAC files)")
        batch = measure("batch", extra=batch_paths, reps=3)
        if batch:
            print(f"  sound_rs load_many  {fmt(batch['median'])}")
        if args.compare and available("soundfile"):
            batch_sf = measure("batch_sf", extra=batch_paths, reps=3)
            if batch_sf:
                print(f"  soundfile loop      {fmt(batch_sf['median'])}")

    print()


if __name__ == "__main__":
    main()

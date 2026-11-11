#!/usr/bin/env python3
"""Benchmark harness: every measurement in its own process.

Peak RSS is process-wide, so timing and memory cannot share a process without
polluting each other. Each worker prints one JSON line and exits.

    python bench/run.py                        # full report
    python bench/run.py --compare              # + competitors
    python bench/run.py --no-audio             # decode-only, no sound devices
    python bench/run.py --reps 15 --report bench/REPORT.md
    python bench/run.py --baseline bench/results-0.2.2.json
    python bench/run.py --python /path/to/venv/bin/python
"""

import argparse
import json
import math
import os
import platform
import random
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
REPS = 10
WARMUP = 1
WORKER_TIMEOUT = 300.0


# --------------------------------------------------------------------------
# statistics


def percentile(values, p):
    """Linear-interpolated percentile, p in 0..100."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_ci(values, confidence=0.95, resamples=2000, seed=1234):
    """Percentile bootstrap CI for the median."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    tail = (1 - confidence) / 2
    return percentile(medians, tail * 100), percentile(medians, (1 - tail) * 100)


def compare(base, new):
    """Relative change of `new` against `base` (both stats dicts), signed %."""
    if not base or not base.get("median"):
        return "n/a"
    delta = (new["median"] - base["median"]) / base["median"] * 100.0
    return f"{delta:+.1f}%"


def summarise(seconds, cpu=None, peak=None, extras=None):
    stamps = sorted(seconds)
    low, high = bootstrap_ci(stamps)
    stats = {
        "runs": len(stamps),
        "min": stamps[0],
        "median": statistics.median(stamps),
        "p95": percentile(stamps, 95),
        "p99": percentile(stamps, 99),
        "ci_low": low,
        "ci_high": high,
    }
    if cpu is not None:
        stats["cpu"] = statistics.median(cpu)
    if peak is not None:
        stats["peak_mb"] = max(peak)
    if extras:
        stats.update(extras)
    return stats


# --------------------------------------------------------------------------
# corpus


def build_corpus(timeout=WORKER_TIMEOUT):
    os.makedirs(CORPUS, exist_ok=True)
    import math as _math
    import wave

    wavs = {}
    for name, seconds in SIZES.items():
        path = os.path.join(CORPUS, f"{name}.wav")
        if not os.path.exists(path):
            body = bytearray()
            for i in range(int(seconds * RATE)):
                value = int(12000 * _math.sin(2 * 3.14159265 * 440 * i / RATE))
                body += value.to_bytes(2, "little", signed=True)
                body += value.to_bytes(2, "little", signed=True)
            with wave.open(path, "w") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(RATE)
                handle.writeframes(bytes(body))
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
                        timeout=timeout,
                    )
                files[f"{name}.{ext}"] = out
    return files


# --------------------------------------------------------------------------
# workers: run in a child process, print one JSON line


def worker(kind, target, extra):
    start_wall = time.perf_counter()
    start_cpu = os.times()
    result = {}

    if kind == "import":
        __import__(target)
    elif kind == "metadata":
        import sound_rs

        sound_rs.get_audio_info(target)
    elif kind == "metadata_sf":
        import soundfile

        soundfile.info(target)
    elif kind == "decode":
        import sound_rs

        sound_rs.load(target)
    elif kind == "decode_sf":
        import soundfile

        soundfile.read(target, dtype="float32")
    elif kind == "pcm":
        import numpy as np
        import sound_rs
        import soundfile

        reference, _ = soundfile.read(target, dtype="float32", always_2d=True)
        ours, _ = sound_rs.load(target)
        n = min(len(ours), len(reference))
        span = min(n, 200_000)
        direct = float(np.abs(ours[:n][:span] - reference[:n][:span]).max())
        best_lag, best_diff = 0, direct
        for lag in range(-96, 97):
            a = ours[max(0, lag) : max(0, lag) + span, 0]
            b = reference[max(0, -lag) : max(0, -lag) + span, 0]
            width = min(len(a), len(b))
            if width < 100:
                continue
            diff = float(np.abs(a[:width] - b[:width]).max())
            if diff < best_diff:
                best_lag, best_diff = lag, diff
        result.update(
            {
                "max_diff": direct,
                "lag": best_lag,
                "aligned_diff": best_diff,
                "frames": int(len(ours)),
                "reference_frames": int(len(reference)),
            }
        )
    elif kind == "play_overhead":
        import numpy as np
        import sound_rs

        seconds = float(extra)
        samples = np.sin(2 * np.pi * 440 * np.arange(int(seconds * RATE)) / RATE).astype(
            np.float32
        )
        sound_rs.play_array(samples, RATE)
    elif kind == "play_overhead_player":
        import numpy as np
        import sound_rs

        seconds = float(extra)
        samples = np.sin(2 * np.pi * 440 * np.arange(int(seconds * RATE)) / RATE).astype(
            np.float32
        )
        player = sound_rs.Player()
        player.play_array(np.zeros(2400, dtype=np.float32), RATE)
        player.wait()
        tick = time.perf_counter()
        player.play_array(samples, RATE)
        player.wait()
        result["play_seconds"] = time.perf_counter() - tick
        player.stop()
    elif kind == "play_overhead_sd":
        import numpy as np
        import sounddevice as sd

        seconds = float(extra)
        samples = np.sin(2 * np.pi * 440 * np.arange(int(seconds * RATE)) / RATE).astype(
            np.float32
        )
        sd.play(samples, RATE)
        sd.wait()
    elif kind == "first_sound":
        import sound_rs

        player = sound_rs.Player()
        tick = time.perf_counter()
        player.play_file(target)
        while player.position <= 0.0:
            time.sleep(0.001)
        result["first_sound"] = time.perf_counter() - tick
        player.stop()
    elif kind == "first_sound_sf":
        import sounddevice as sd
        import soundfile

        tick = time.perf_counter()
        data, rate = soundfile.read(target, dtype="float32")
        sd.play(data, rate)
        while True:
            if sd.get_stream().time > 0:
                break
            time.sleep(0.001)
        result["first_sound"] = time.perf_counter() - tick
        sd.stop()
    elif kind == "batch":
        import sound_rs

        paths = extra.split(",")
        sound_rs.load_many(paths)
        result["files"] = len(paths)
    elif kind == "batch_sf":
        import soundfile

        paths = extra.split(",")
        for path in paths:
            soundfile.read(path, dtype="float32")
        result["files"] = len(paths)
    elif kind == "stream":
        import sound_rs

        seconds = float(extra)
        player = sound_rs.Player()
        player.play_file(target)
        tick = time.perf_counter()
        time.sleep(seconds)
        played = player.position
        wall = time.perf_counter() - tick
        result["rtf"] = played / wall if wall else 0.0
        result["underruns"] = getattr(player, "underruns", 0)
        player.stop()
    elif kind == "batch_scaling":
        import sound_rs

        paths = extra.split(",")
        counts = sorted({1, 2, 4, 8, min(8, os.cpu_count() or 1)})
        scaling = {}
        for threads in counts:
            tick = time.perf_counter()
            sound_rs.load_many(paths, threads=threads)
            scaling[threads] = time.perf_counter() - tick
        result["scaling"] = scaling
    elif kind == "soak":
        import numpy as np
        import sound_rs

        samples = np.zeros(int(0.5 * RATE), dtype=np.float32)
        peak = []
        for _ in range(int(extra)):
            sound_rs.play_array(samples, RATE)
            peak.append(resource_peak_mb())
        result["rss_start"] = peak[0]
        result["rss_end"] = peak[-1]
        result["rss_growth"] = peak[-1] - peak[0]
    else:
        raise SystemExit(f"unknown worker {kind}")

    wall = time.perf_counter() - start_wall
    end_cpu = os.times()
    result.setdefault("seconds", wall)
    result["cpu"] = (end_cpu.user - start_cpu.user) + (
        end_cpu.system - start_cpu.system
    )
    peak_mb = resource_peak_mb()
    if peak_mb is not None:
        result["peak_mb"] = peak_mb
    print(json.dumps(result))


def resource_peak_mb():
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def spawn(kind, target="", extra="", python=None, timeout=WORKER_TIMEOUT):
    command = [python or sys.executable, os.path.abspath(__file__), "--worker", kind, target]
    if extra:
        command.append(extra)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as error:
        print(f"benchmark {kind}: {error}", file=sys.stderr)
        return None
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        print(f"benchmark {kind}: worker failed ({completed.returncode})"
              + (f": {detail[-1]}" if detail else ""), file=sys.stderr)
        return None
    lines = completed.stdout.strip().splitlines()
    try:
        result = json.loads(lines[-1]) if lines else None
        if not isinstance(result, dict) or not isinstance(result.get("seconds"), (int, float)):
            raise ValueError("missing numeric seconds")
        if not math.isfinite(result["seconds"]) or result["seconds"] < 0:
            raise ValueError("invalid worker timing")
        return result
    except (ValueError, TypeError) as error:
        print(f"benchmark {kind}: invalid worker output ({error})", file=sys.stderr)
        return None


def measure(kind, target="", extra="", reps=REPS, python=None, timeout=WORKER_TIMEOUT):
    """Warmup, then all requested runs; a failed run invalidates the summary."""
    if reps < 1:
        raise ValueError("reps must be positive")
    for _ in range(WARMUP):
        if spawn(kind, target, extra, python, timeout) is None:
            return None
    runs = []
    for _ in range(reps):
        run = spawn(kind, target, extra, python, timeout)
        if run is None:
            return None
        runs.append(run)

    seconds = [run["seconds"] for run in runs]
    cpu = [run["cpu"] for run in runs if "cpu" in run]
    peak = [run["peak_mb"] for run in runs if run.get("peak_mb") is not None]
    extras = {
        key: value
        for key, value in runs[-1].items()
        if key not in {"seconds", "cpu", "peak_mb"}
    }
    return summarise(seconds, cpu or None, peak or None, extras)


# --------------------------------------------------------------------------
# report


def fmt_seconds(stats, key="median"):
    value = stats.get(key)
    if value is None:
        return "-"
    if value < 0.01:
        return f"{value * 1e6:.0f} µs"
    if value < 1:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.3f} s"


def line(name, variant, stats, baseline=None):
    base = (baseline or {}).get(name, {}).get(variant)
    delta = compare(base, stats) if base else ""
    peak = stats.get("peak_mb")
    cpu = stats.get("cpu")
    parts = [
        f"{fmt_seconds(stats):>9}",
        f"p95 {fmt_seconds(stats, 'p95'):>9}",
    ]
    if cpu is not None:
        parts.append(f"cpu {cpu:.2f}s")
    if peak is not None:
        parts.append(f"rss {peak:.0f}MB")
    if delta:
        parts.append(f"{delta} vs base")
    print(f"  {variant:<12} " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--compare", action="store_true", help="also measure competitors")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--no-audio", action="store_true", help="skip every playback, streaming and soak scenario")
    parser.add_argument("--timeout", type=float, default=WORKER_TIMEOUT, help="maximum seconds per worker or corpus encoder (default: 300)")
    parser.add_argument("--report", help="write a markdown report to this path")
    parser.add_argument("--baseline", help="compare against a previous results.json")
    parser.add_argument("--python", help="interpreter to run workers with")
    parser.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")

    if args.worker:
        kind, *rest = args.worker
        target = rest[0] if rest else ""
        extra = rest[1] if len(rest) > 1 else ""
        worker(kind, target, extra)
        return

    import importlib.util

    def available(module):
        return importlib.util.find_spec(module) is not None

    baseline = None
    if args.baseline:
        with open(args.baseline) as handle:
            baseline = json.load(handle)["scenarios"]

    files = build_corpus(timeout=args.timeout)
    machine = platform.processor() or platform.machine()
    print(
        f"\nsound_rs benchmark — {machine}, {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}, reps={args.reps}"
    )
    results = {}

    def record(name, variant, stats):
        results.setdefault(name, {})[variant] = stats

    # import
    print("\nimport time (cold process)")
    for module in ["sound_rs", "soundfile", "sounddevice"]:
        if args.no_audio and module == "sounddevice":
            continue
        if module != "sound_rs" and not (args.compare and available(module)):
            continue
        stats = measure("import", module, reps=max(3, args.reps), python=args.python, timeout=args.timeout)
        if stats:
            record("import", module, stats)
            line("import", module, stats, baseline)

    # metadata and decode
    print("\nmetadata / decode")
    for name, path in sorted(files.items()):
        meta = measure("metadata", path, reps=args.reps, python=args.python, timeout=args.timeout)
        dec = measure("decode", path, reps=args.reps, python=args.python, timeout=args.timeout)
        if dec:
            record(f"decode:{name}", "sound_rs", dec)
            line(f"decode:{name}", name, dec, baseline)
        if meta:
            record(f"metadata:{name}", "sound_rs", meta)

        if args.compare:
            if available("soundfile"):
                sf_meta = measure("metadata_sf", path, reps=args.reps, python=args.python, timeout=args.timeout)
                sf_dec = measure("decode_sf", path, reps=args.reps, python=args.python, timeout=args.timeout)
                if sf_meta:
                    record(f"metadata:{name}", "soundfile", sf_meta)
                if sf_dec:
                    record(f"decode:{name}", "soundfile", sf_dec)
                    line(f"decode:{name}", "soundfile", sf_dec)
            if dec:
                pcm = spawn("pcm", path, python=args.python, timeout=args.timeout)
                if pcm:
                    detail = (
                        "identical"
                        if pcm["aligned_diff"] == 0
                        else f"diff {pcm['aligned_diff']:.1e}"
                    )
                    if pcm["lag"]:
                        detail += f", offset {pcm['lag']} samples"
                    print(f"  {'':<12} PCM {detail}  frames {pcm['frames']}")

    if args.no_audio:
        print("\naudio scenarios skipped (--no-audio)")
    else:
        # playback
        print("\nplayback, 3 s tone")
        for kind, variant in [
            ("play_overhead", "one-shot"),
            ("play_overhead_player", "player"),
        ]:
            stats = measure(kind, extra="3.0", reps=3, python=args.python, timeout=args.timeout)
            if stats:
                record("playback", variant, stats)
                overhead = stats.get("play_seconds", stats["median"]) - 3.0
                print(
                    f"  {variant:<12} {fmt_seconds(stats):>9}  overhead {overhead:+.3f} s"
                    f"  p95 {fmt_seconds(stats, 'p95')}"
                )
        if args.compare and available("sounddevice"):
            stats = measure("play_overhead_sd", extra="3.0", reps=3, python=args.python, timeout=args.timeout)
            if stats:
                record("playback", "sounddevice", stats)
                print(
                    f"  {'sounddevice':<12} {fmt_seconds(stats):>9}"
                    f"  overhead {stats['median'] - 3.0:+.3f} s"
                    f"  p95 {fmt_seconds(stats, 'p95')}"
                )

        # streaming
        long_path = files.get("long.flac") or files.get("long")
        if long_path:
            print(f"\nstreaming {os.path.basename(long_path)}")
            first = measure("first_sound", long_path, reps=3, python=args.python, timeout=args.timeout)
            if first:
                record("first_sound", "sound_rs", first)
                print(
                    f"  {'player':<12} first sound {fmt_seconds(first, 'first_sound'):>9}"
                    f"  rss {first.get('peak_mb', 0):.0f}MB"
                )
            stream = measure("stream", long_path, extra="5", reps=3, python=args.python, timeout=args.timeout)
            if stream:
                record("stream", "sound_rs", stream)
                print(
                    f"  {'rtf':<12} {stream.get('rtf', 0):.2f}x realtime"
                    f"  underruns {stream.get('underruns', 0)}"
                )
            if args.compare and available("sounddevice") and available("soundfile"):
                first_sf = measure("first_sound_sf", long_path, reps=3, python=args.python, timeout=args.timeout)
                if first_sf:
                    record("first_sound", "soundfile+sounddevice", first_sf)
                    print(
                        f"  {'soundfile+sd':<12} first sound"
                        f" {fmt_seconds(first_sf, 'first_sound'):>9}"
                        f"  rss {first_sf.get('peak_mb', 0):.0f}MB"
                    )

    # batch
    flacs = ",".join(p for n, p in sorted(files.items()) if n.endswith(".flac"))
    if flacs:
        print(f"\nbatch decode ({len(flacs.split(','))} FLAC files)")
        batch = measure("batch", extra=flacs, reps=3, python=args.python, timeout=args.timeout)
        if batch:
            record("batch", "sound_rs", batch)
            print(f"  {'load_many':<12} {fmt_seconds(batch)}")
        if args.compare and available("soundfile"):
            batch_sf = measure("batch_sf", extra=flacs, reps=3, python=args.python, timeout=args.timeout)
            if batch_sf:
                record("batch", "soundfile", batch_sf)
                print(f"  {'soundfile':<12} {fmt_seconds(batch_sf)}")
        scaling = measure("batch_scaling", extra=flacs, reps=1, python=args.python, timeout=args.timeout)
        if scaling and "scaling" in scaling:
            parts = [
                f"{threads} thr {seconds * 1000:.0f} ms"
                for threads, seconds in sorted(scaling["scaling"].items())
            ]
            print("  scaling      " + ", ".join(parts))

    if not args.no_audio:
        # soak
        soak = measure("soak", extra="200", reps=1, python=args.python, timeout=args.timeout)
        if soak:
            record("soak", "sound_rs", soak)
            print(
                f"\nsoak          200 play cycles, RSS {soak.get('rss_start', 0):.0f}"
                f" -> {soak.get('rss_end', 0):.0f} MB"
                f" (growth {soak.get('rss_growth', 0):+.1f} MB)"
            )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine": machine,
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "reps": args.reps,
            "audio_enabled": not args.no_audio,
            "worker_timeout": args.timeout,
            "skipped_scenarios": ["playback", "first_sound", "stream", "soak"] if args.no_audio else [],
        },
        "scenarios": results,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

    if args.report:
        with open(args.report, "w") as handle:
            handle.write("# sound_rs benchmark\n\n")
            handle.write(
                f"{machine}, {payload['meta']['system']}, Python "
                f"{payload['meta']['python']}, reps={args.reps}, "
                f"{payload['meta']['timestamp']}\n\n"
            )
            for name in sorted(results):
                handle.write(f"## {name}\n\n")
                handle.write("| variant | median | p95 | p99 | cpu | rss |\n")
                handle.write("|---|---|---|---|---|---|\n")
                for variant, stats in sorted(results[name].items()):
                    handle.write(
                        f"| {variant} | {fmt_seconds(stats)} | "
                        f"{fmt_seconds(stats, 'p95')} | {fmt_seconds(stats, 'p99')} | "
                        f"{stats.get('cpu', 0):.2f}s | {stats.get('peak_mb', 0):.0f}MB |\n"
                    )
                handle.write("\n")
        print(f"\nreport written to {args.report}")

    print(f"results written to {args.out}\n")


if __name__ == "__main__":
    main()

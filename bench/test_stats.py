#!/usr/bin/env python3
"""Checks for the benchmark statistics. Plain asserts, no test framework.

    python3 bench/test_stats.py
"""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run as benchmark  # noqa: E402
from run import bootstrap_ci, compare, percentile, summarise  # noqa: E402


def test_percentile_edges():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50.5, percentile(values, 50)
    assert percentile(values, 99) >= 99, percentile(values, 99)
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 100
    assert percentile([7.0], 95) == 7.0


def test_bootstrap_ci_brackets_the_median():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    low, high = bootstrap_ci(values, resamples=500)
    median = 5.5
    assert low <= median <= high, (low, median, high)
    assert high - low < 10.0, (low, high)


def test_compare_reports_slower_and_faster():
    assert compare({"median": 1.0}, {"median": 1.25}) == "+25.0%"
    assert compare({"median": 1.0}, {"median": 0.8}) == "-20.0%"
    assert compare(None, {"median": 1.0}) == "n/a"


def test_summarise_has_the_promised_keys():
    stats = summarise([0.1, 0.2, 0.3], cpu=[0.05, 0.06], peak=[10.0, 12.0])
    for key in ("median", "p95", "p99", "ci_low", "ci_high", "cpu", "peak_mb"):
        assert key in stats, key
    assert stats["peak_mb"] == 12.0


def test_worker_timeout_is_bounded_and_reported():
    with mock.patch.object(benchmark.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 0.1)) as run, \
            contextlib.redirect_stderr(io.StringIO()) as errors:
        assert benchmark.spawn("decode", "fixture.wav", timeout=0.1) is None
    assert run.call_args.kwargs["timeout"] == 0.1
    assert "timed out" in errors.getvalue()


def test_worker_rejects_empty_or_invalid_measurements():
    for output in ("", "not json", "[]", "{}", '{"seconds": -1}', '{"seconds": NaN}'):
        completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with mock.patch.object(benchmark.subprocess, "run", return_value=completed), \
                contextlib.redirect_stderr(io.StringIO()):
            assert benchmark.spawn("decode") is None, output


def test_failed_measured_run_invalidates_summary():
    valid = {"seconds": 0.1, "cpu": 0.05}
    with mock.patch.object(benchmark, "spawn", side_effect=[valid, valid, None]):
        assert benchmark.measure("decode", reps=2) is None


def test_no_audio_skips_every_device_scenario():
    calls = []
    stats = summarise([0.01, 0.02, 0.03])

    def measure(kind, target="", **kwargs):
        calls.append((kind, target))
        assert kwargs["timeout"] == 7.0
        return stats

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "results.json"
        with mock.patch.object(sys, "argv", ["run.py", "--no-audio", "--compare", "--timeout", "7", "--out", str(output)]), \
                mock.patch.object(benchmark, "build_corpus", return_value={"long.flac": "fixture.flac", "short": "fixture.wav"}), \
                mock.patch.object(benchmark, "measure", side_effect=measure), \
                mock.patch.object(benchmark, "spawn", return_value=None), \
                mock.patch("importlib.util.find_spec", return_value=object()), \
                contextlib.redirect_stdout(io.StringIO()):
            benchmark.main()
        payload = json.loads(output.read_text())
    forbidden = {"play_overhead", "play_overhead_player", "play_overhead_sd", "first_sound", "first_sound_sf", "stream", "soak"}
    assert not (forbidden & {kind for kind, _ in calls}), calls
    assert ("import", "sounddevice") not in calls
    assert payload["meta"]["audio_enabled"] is False
    assert set(payload["meta"]["skipped_scenarios"]) == {"playback", "first_sound", "stream", "soak"}
    assert "decode:long.flac" in payload["scenarios"]
    assert "batch" in payload["scenarios"]


def test_invalid_benchmark_arguments_fail_before_work():
    for argument in (["--reps", "0"], ["--timeout", "0"], ["--timeout", "nan"]):
        with mock.patch.object(sys, "argv", ["run.py", *argument]), \
                mock.patch.object(benchmark, "build_corpus") as build, \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                benchmark.main()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"accepted {argument}")
        build.assert_not_called()


if __name__ == "__main__":
    for name, test in sorted(list(globals().items())):
        if name.startswith("test_") and callable(test):
            test()
    print("bench stats and harness checks ok")

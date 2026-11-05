#!/usr/bin/env python3
"""Checks for the benchmark statistics. Plain asserts, no test framework.

    python3 bench/test_stats.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


if __name__ == "__main__":
    test_percentile_edges()
    test_bootstrap_ci_brackets_the_median()
    test_compare_reports_slower_and_faster()
    test_summarise_has_the_promised_keys()
    print("bench stats ok")

"""Time the complete public largest-rectangle functions."""

import contextlib
import csv
import io
import statistics
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

from benchmark_largest_rectangle import (  # noqa: E402
    load_notebook_functions,
    load_saved_results,
    make_synthetic_results,
)


OUTPUT = BENCH_DIR / "largest_rectangle_end_to_end.csv"
KP_MIN = -10.0
KP_MAX = 10.0
WARMUP_RUNS = 2
TIMED_REPETITIONS = 20
SYNTHETIC_SHAPES = (
    (32, 64),
    (64, 128),
    (128, 256),
    (128, 512),
    (256, 256),
    (256, 512),
)


def same_output(left, right):
    if left is None or right is None:
        return left is None and right is None
    return all(
        left[field] == right[field]
        for field in ("Kp_range", "Ki_range", "area", "center")
    )


def call(function, results, resolution, max_ki_gap):
    return function(
        results,
        KP_MIN,
        KP_MAX,
        plot_resolution=resolution,
        max_ki_gap=max_ki_gap,
    )


def make_cases():
    saved_results = load_saved_results()
    cases = [
        (
            f"{plant}_cached",
            "saved benchmark intervals",
            saved_results[plant],
            300,
            0.15,
        )
        for plant in ("G1", "G5", "G6")
    ]

    for seed_offset, (height, width) in enumerate(SYNTHETIC_SHAPES):
        _, results, max_gap = make_synthetic_results(
            height, width, seed=20261124 + seed_offset
        )
        cases.append(
            (
                f"synthetic_{height}x{width}",
                "fixed-seed synthetic intervals",
                results,
                width,
                max_gap,
            )
        )
    return cases


def benchmark_case(case, reference_function, optimized_function):
    name, source, results, resolution, max_ki_gap = case
    reference_baseline = call(
        reference_function, results, resolution, max_ki_gap
    )
    optimized_baseline = call(
        optimized_function, results, resolution, max_ki_gap
    )
    if not same_output(reference_baseline, optimized_baseline):
        raise AssertionError(f"Public function results differ for {name}.")

    for _ in range(WARMUP_RUNS):
        reference_output = call(
            reference_function, results, resolution, max_ki_gap
        )
        optimized_output = call(
            optimized_function, results, resolution, max_ki_gap
        )
        if not (
            same_output(reference_baseline, reference_output)
            and same_output(optimized_baseline, optimized_output)
        ):
            raise AssertionError(f"Warm-up output changed for {name}.")

    reference_times = []
    optimized_times = []
    with contextlib.redirect_stdout(io.StringIO()):
        for repetition in range(TIMED_REPETITIONS):
            order = (
                (
                    (reference_function, reference_times, reference_baseline),
                    (optimized_function, optimized_times, optimized_baseline),
                )
                if repetition % 2 == 0
                else (
                    (optimized_function, optimized_times, optimized_baseline),
                    (reference_function, reference_times, reference_baseline),
                )
            )
            for function, timings, expected in order:
                started = time.perf_counter()
                result = call(function, results, resolution, max_ki_gap)
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                if not same_output(expected, result):
                    raise AssertionError(
                        f"Timed output changed for {name}."
                    )

    reference_mean = statistics.mean(reference_times)
    optimized_mean = statistics.mean(optimized_times)
    return {
        "case": name,
        "source": source,
        "H": len(results),
        "W": resolution,
        "max_ki_gap": max_ki_gap,
        "warmup_runs_per_function": WARMUP_RUNS,
        "timed_repetitions_per_function": TIMED_REPETITIONS,
        "reference_mean_seconds": reference_mean,
        "reference_median_seconds": statistics.median(reference_times),
        "reference_std_seconds": statistics.stdev(reference_times),
        "optimized_mean_seconds": optimized_mean,
        "optimized_median_seconds": statistics.median(optimized_times),
        "optimized_std_seconds": statistics.stdev(optimized_times),
        "mean_runtime_speedup": reference_mean / optimized_mean,
        "median_runtime_speedup": (
            statistics.median(reference_times)
            / statistics.median(optimized_times)
        ),
        "reference_area": (
            "" if reference_baseline is None else reference_baseline["area"]
        ),
        "optimized_area": (
            "" if optimized_baseline is None else optimized_baseline["area"]
        ),
        "all_returned_fields_match": True,
    }


def main():
    reference_function, optimized_function, definition_cell, call_cell = (
        load_notebook_functions()
    )
    cases = make_cases()
    rows = [
        benchmark_case(case, reference_function, optimized_function)
        for case in cases
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Notebook definition cell: {definition_cell}")
    print(f"Notebook caller cell: {call_cell}")
    print(f"Results written to: {OUTPUT}")
    for row in rows:
        print(
            f"{row['case']}: H={row['H']} W={row['W']} "
            f"reference={row['reference_mean_seconds']:.6g}s "
            f"optimized={row['optimized_mean_seconds']:.6g}s "
            f"speedup={row['mean_runtime_speedup']:.2f}x "
            f"equivalent={row['all_returned_fields_match']}"
        )


if __name__ == "__main__":
    main()

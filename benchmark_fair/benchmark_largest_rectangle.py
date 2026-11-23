"""Benchmark largest-rectangle search kernels on cached and synthetic inputs."""

import contextlib
import csv
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from largest_stable_rectangle import (
    find_largest_stable_rectangle as optimized_function,
)


OUTPUT = Path(__file__).resolve().parent / "largest_rectangle_performance.csv"
KI_ACCOUNTING = Path(__file__).resolve().parent / "ki_row_accounting.csv"
NOTEBOOK = ROOT / "stability_without_poles.ipynb"
REPETITIONS = 10
KP_MIN = -10.0
KP_MAX = 10.0


def load_notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    reference_cell_index = None
    caller_cell_index = None
    reference_source = None

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def find_largest_stable_rectangle_reference(" in source:
            reference_cell_index = cell_index
            reference_source = source
        if "rectangle = find_largest_stable_rectangle(" in source:
            caller_cell_index = cell_index

    if reference_cell_index is None or caller_cell_index is None:
        raise RuntimeError("Notebook rectangle definition or call was not found.")
    if reference_cell_index >= caller_cell_index:
        raise RuntimeError("Notebook calls the rectangle function before its definition.")
    if "from largest_stable_rectangle import find_largest_stable_rectangle" not in reference_source:
        raise RuntimeError("Notebook does not import the optimized implementation.")

    namespace = {"__name__": "__main__", "np": np}
    exec(
        compile(reference_source, f"{NOTEBOOK.name}:cell{reference_cell_index}", "exec"),
        namespace,
    )
    return (
        namespace["find_largest_stable_rectangle_reference"],
        namespace["find_largest_stable_rectangle"],
        reference_cell_index,
        caller_cell_index,
    )


def load_saved_results():
    selections = {
        "G1": ("repeated_G1", 1),
        "G5": ("multi_plant", 1),
        "G6": ("multi_plant", 1),
    }
    results_by_plant = {plant: {} for plant in selections}

    with KI_ACCOUNTING.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            plant = row["plant"].split(":", 1)[0]
            if plant not in selections:
                continue
            run_group, repeat = selections[plant]
            if (
                row["run_group"] != run_group
                or int(row["repeat"]) != repeat
                or row["strategy"] != "exhaustive"
                or row["retained_in_original_results"].lower() != "true"
            ):
                continue
            intervals = json.loads(row["intervals"])
            results_by_plant[plant][float(row["Ki"])] = [
                (float(left), float(right)) for left, right in intervals
            ]

    for plant, results in results_by_plant.items():
        if not results:
            raise RuntimeError(f"No cached stable Ki rows found for {plant}.")
    return results_by_plant


def make_synthetic_results(height, width, seed):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((height, width), dtype=bool)

    for row in range(height):
        interval_count = int(rng.integers(1, 5))
        for _ in range(interval_count):
            left = int(rng.integers(0, width))
            right = int(rng.integers(left, width))
            matrix[row, left : right + 1] = True

    ki_values = np.arange(height, dtype=float) * 0.05
    results = matrix_to_results(matrix, ki_values, KP_MIN, KP_MAX)
    return matrix, results, 0.075


def matrix_to_results(matrix, ki_values, kp_min, kp_max):
    width = matrix.shape[1]
    kp_values = np.linspace(kp_min, kp_max, width)
    results = {}

    for ki, row in zip(ki_values, matrix):
        intervals = []
        column = 0
        while column < width:
            if not row[column]:
                column += 1
                continue
            start = column
            while column + 1 < width and row[column + 1]:
                column += 1
            intervals.append((float(kp_values[start]), float(kp_values[column])))
            column += 1
        results[float(ki)] = intervals

    return results


def build_stability_matrix(results, kp_values, ki_values):
    stable_map = np.zeros((len(ki_values), len(kp_values)), dtype=bool)
    for row_index, ki in enumerate(ki_values):
        for left, right in results[ki]:
            stable_map[row_index, (kp_values >= left) & (kp_values <= right)] = True
    return stable_map


def reference_search_kernel(stable_map, kp_values, ki_values, max_ki_gap):
    best_area = 0
    best_rectangle = None

    for top in range(len(ki_values) - 1):
        common = stable_map[top].copy()
        for bottom in range(top + 1, len(ki_values)):
            if ki_values[bottom] - ki_values[bottom - 1] > max_ki_gap:
                break

            common &= stable_map[bottom]
            changes = np.diff(np.r_[False, common, False].astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            for start, end in zip(starts, ends):
                kp_left = kp_values[start]
                kp_right = kp_values[end - 1]
                ki_low = ki_values[top]
                ki_high = ki_values[bottom]
                area = (kp_right - kp_left) * (ki_high - ki_low)

                if area > best_area:
                    best_area = area
                    best_rectangle = {
                        "Kp_range": (kp_left, kp_right),
                        "Ki_range": (ki_low, ki_high),
                        "area": area,
                        "center": (
                            (kp_left + kp_right) / 2,
                            (ki_low + ki_high) / 2,
                        ),
                    }

    return best_rectangle


def optimized_search_kernel(stable_map, kp_values, ki_values, max_ki_gap):
    best_area = 0
    best_rectangle = None
    best_order = None
    heights = np.zeros(len(kp_values), dtype=float)
    start_rows = np.full(len(kp_values), -1, dtype=int)

    for bottom in range(len(ki_values)):
        if (
            bottom > 0
            and ki_values[bottom] - ki_values[bottom - 1] > max_ki_gap
        ):
            heights.fill(0)
            start_rows.fill(-1)

        stable_row = stable_map[bottom]
        for column in range(len(kp_values)):
            if not stable_row[column]:
                heights[column] = 0
                start_rows[column] = -1
                continue

            if start_rows[column] == -1:
                start_rows[column] = bottom
            heights[column] = ki_values[bottom] - ki_values[start_rows[column]]

        stack = []
        for column in range(len(kp_values) + 1):
            current_height = (
                heights[column] if column < len(kp_values) else -np.inf
            )
            while stack and heights[stack[-1]] > current_height:
                bar = stack.pop()
                left = stack[-1] + 1 if stack else 0
                right = column - 1
                top = start_rows[bar]
                if top < 0 or top >= bottom or left >= right:
                    continue

                kp_left = kp_values[left]
                kp_right = kp_values[right]
                ki_low = ki_values[top]
                ki_high = ki_values[bottom]
                area = (kp_right - kp_left) * (ki_high - ki_low)
                order = (top, bottom, left, right)

                if (
                    area > best_area
                    or (
                        area == best_area
                        and best_order is not None
                        and order < best_order
                    )
                ):
                    best_area = area
                    best_order = order
                    best_rectangle = {
                        "Kp_range": (kp_left, kp_right),
                        "Ki_range": (ki_low, ki_high),
                        "area": area,
                        "center": (
                            (kp_left + kp_right) / 2,
                            (ki_low + ki_high) / 2,
                        ),
                    }

            if column < len(kp_values):
                stack.append(column)

    return best_rectangle


def outputs_match(first, second):
    if first is None or second is None:
        return first is None and second is None
    return (
        first["Kp_range"] == second["Kp_range"]
        and first["Ki_range"] == second["Ki_range"]
        and first["area"] == second["area"]
        and first["center"] == second["center"]
    )


def call_quietly(function, results, resolution, max_ki_gap):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(
            results,
            KP_MIN,
            KP_MAX,
            plot_resolution=resolution,
            max_ki_gap=max_ki_gap,
        )


def make_cases(saved_results):
    cases = []
    for plant in ("G1", "G5", "G6"):
        cases.append(
            (
                f"{plant}_cached",
                "saved benchmark intervals",
                saved_results[plant],
                300,
                0.15,
            )
        )

    for index, (height, width) in enumerate(
        (
            (32, 64),
            (64, 128),
            (128, 256),
            (128, 512),
            (256, 256),
            (256, 512),
        )
    ):
        _, results, max_gap = make_synthetic_results(
            height, width, seed=20261124 + index
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


def benchmark_case(
    case_name,
    source,
    results,
    resolution,
    max_ki_gap,
    reference_function,
):
    ki_values = np.array(sorted(results.keys()), dtype=float)
    kp_values = np.linspace(KP_MIN, KP_MAX, resolution)
    stable_map = build_stability_matrix(results, kp_values, ki_values)

    # Validate that the exact public APIs and the isolated search kernels agree.
    reference_api = call_quietly(
        reference_function, results, resolution, max_ki_gap
    )
    optimized_api = call_quietly(
        optimized_function, results, resolution, max_ki_gap
    )
    reference_kernel = reference_search_kernel(
        stable_map, kp_values, ki_values, max_ki_gap
    )
    optimized_kernel = optimized_search_kernel(
        stable_map, kp_values, ki_values, max_ki_gap
    )
    if not (
        outputs_match(reference_api, optimized_api)
        and outputs_match(reference_api, reference_kernel)
        and outputs_match(optimized_api, optimized_kernel)
    ):
        raise AssertionError(f"Output mismatch in {case_name}.")

    reference_search_kernel(stable_map, kp_values, ki_values, max_ki_gap)
    optimized_search_kernel(stable_map, kp_values, ki_values, max_ki_gap)

    reference_times = []
    optimized_times = []
    for repetition in range(REPETITIONS):
        order = (
            (
                ("reference", reference_search_kernel, reference_times),
                ("optimized", optimized_search_kernel, optimized_times),
            )
            if repetition % 2 == 0
            else (
                ("optimized", optimized_search_kernel, optimized_times),
                ("reference", reference_search_kernel, reference_times),
            )
        )
        for _, function, timings in order:
            started = time.perf_counter()
            function(stable_map, kp_values, ki_values, max_ki_gap)
            timings.append(time.perf_counter() - started)

    reference_area = (
        "" if reference_api is None else float(reference_api["area"])
    )
    optimized_area = (
        "" if optimized_api is None else float(optimized_api["area"])
    )
    mean_reference = statistics.mean(reference_times)
    mean_optimized = statistics.mean(optimized_times)

    return {
        "case": case_name,
        "source": source,
        "H": len(ki_values),
        "W": len(kp_values),
        "max_ki_gap": max_ki_gap,
        "timed_repetitions": REPETITIONS,
        "reference_mean_seconds": mean_reference,
        "reference_median_seconds": statistics.median(reference_times),
        "reference_std_seconds": statistics.stdev(reference_times),
        "optimized_mean_seconds": mean_optimized,
        "optimized_median_seconds": statistics.median(optimized_times),
        "optimized_std_seconds": statistics.stdev(optimized_times),
        "mean_runtime_speedup": mean_reference / mean_optimized,
        "median_runtime_speedup": (
            statistics.median(reference_times)
            / statistics.median(optimized_times)
        ),
        "reference_area": reference_area,
        "optimized_area": optimized_area,
        "all_returned_fields_match": True,
    }


def main():
    reference_function, notebook_optimized, definition_cell, call_cell = (
        load_notebook_functions()
    )
    if notebook_optimized is not optimized_function:
        raise RuntimeError("Notebook optimized import resolved to a different function.")

    saved_results = load_saved_results()
    cases = make_cases(saved_results)
    rows = [
        benchmark_case(*case, reference_function=reference_function)
        for case in cases
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Notebook definition cell: {definition_cell}")
    print(f"Notebook caller cell: {call_cell}")
    print(f"Benchmark results written to: {OUTPUT}")
    for row in rows:
        print(
            f"{row['case']}: H={row['H']} W={row['W']} "
            f"ref={row['reference_mean_seconds']:.6g}s "
            f"opt={row['optimized_mean_seconds']:.6g}s "
            f"speedup={row['mean_runtime_speedup']:.2f}x "
            f"equivalent={row['all_returned_fields_match']}"
        )


if __name__ == "__main__":
    main()

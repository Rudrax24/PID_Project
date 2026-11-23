import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from largest_stable_rectangle import find_largest_stable_rectangle


NOTEBOOK_PATH = Path(__file__).with_name("stability_without_poles.ipynb")


def _load_reference():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"np": np}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def find_largest_stable_rectangle_reference(" in source:
            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
            return namespace["find_largest_stable_rectangle_reference"]
    raise AssertionError("Reference rectangle function was not found in notebook.")


REFERENCE = _load_reference()


def _call_quietly(function, results, kp_min, kp_max, resolution, max_gap):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(
            results,
            kp_min,
            kp_max,
            plot_resolution=resolution,
            max_ki_gap=max_gap,
        )


def _assert_equivalent(results, kp_min, kp_max, resolution, max_gap):
    reference = _call_quietly(
        REFERENCE, results, kp_min, kp_max, resolution, max_gap
    )
    optimized = _call_quietly(
        find_largest_stable_rectangle,
        results,
        kp_min,
        kp_max,
        resolution,
        max_gap,
    )

    assert (reference is None) == (optimized is None)
    if reference is None:
        return

    assert optimized["Kp_range"] == pytest.approx(reference["Kp_range"])
    assert optimized["Ki_range"] == pytest.approx(reference["Ki_range"])
    assert optimized["area"] == pytest.approx(reference["area"])
    assert optimized["center"] == pytest.approx(reference["center"])


def _matrix_results(matrix, ki_values, kp_min=0.0, kp_max=None):
    matrix = np.asarray(matrix, dtype=bool)
    width = matrix.shape[1]
    if kp_max is None:
        kp_max = float(width - 1)
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
            intervals.append((kp_values[start], kp_values[column]))
            column += 1
        results[float(ki)] = intervals

    return results, kp_min, kp_max, width


@pytest.mark.parametrize(
    ("matrix", "ki_values", "max_gap"),
    [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            [0, 1, 2, 3, 4],
            1.0,
        ),
        (
            np.array(
                [
                    [0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0],
                ],
                dtype=bool,
            ),
            [0, 0.1, 0.25, 0.4, 0.6],
            0.2,
        ),
        (
            np.array(
                [
                    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
                    [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
                ],
                dtype=bool,
            ),
            [0, 0.1, 0.2, 0.3, 0.4],
            0.1,
        ),
        (
            np.array(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                ],
                dtype=bool,
            ),
            [0, 0.049, 0.099, 0.15],
            0.05,
        ),
        (
            np.ones((4, 6), dtype=bool),
            [0, 0.05, 0.1, 0.15],
            0.05,
        ),
        (
            np.ones((4, 6), dtype=bool),
            [0, 0.051, 0.102, 0.153],
            0.05,
        ),
        (
            np.array(
                [
                    [1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                ],
                dtype=bool,
            ),
            [0, 0.05, 0.1],
            0.1,
        ),
        (
            np.array(
                [
                    [1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 0, 1],
                ],
                dtype=bool,
            ),
            [0, 1, 2],
            1,
        ),
        (
            np.array(
                [
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
            [0, 1, 2, 3],
            1,
        ),
    ],
)
def test_optimized_matches_reference_on_structured_matrices(
    matrix, ki_values, max_gap
):
    results, kp_min, kp_max, resolution = _matrix_results(matrix, ki_values)
    _assert_equivalent(results, kp_min, kp_max, resolution, max_gap)


@pytest.mark.parametrize(
    "results",
    [
        {},
        {0.0: [(0.0, 2.0)]},
        {0.0: [(1.0, 1.0)], 1.0: [(1.0, 1.0)]},
        {0.0: [], 1.0: [], 2.0: []},
    ],
)
def test_optimized_matches_reference_on_empty_and_degenerate_inputs(results):
    _assert_equivalent(results, 0.0, 2.0, 3, 1.0)


def test_equal_area_rectangles_keep_reference_first_wins_order():
    matrix = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    results, kp_min, kp_max, resolution = _matrix_results(
        matrix, [0, 1, 2, 3]
    )
    _assert_equivalent(results, kp_min, kp_max, resolution, 1)
    result = _call_quietly(
        find_largest_stable_rectangle,
        results,
        kp_min,
        kp_max,
        resolution,
        1,
    )
    assert result["Kp_range"] == (0.0, 2.0)
    assert result["Ki_range"] == (0.0, 1.0)


def test_one_kp_column_has_no_positive_area_rectangle():
    results = {
        0.0: [(0.0, 2.0)],
        1.0: [(0.0, 2.0)],
        2.0: [(0.0, 2.0)],
    }
    _assert_equivalent(results, 0.0, 2.0, 1, 1.0)


def test_randomized_matrices_match_reference():
    rng = np.random.default_rng(20261124)
    increments_to_sample = np.array([0.025, 0.05, 0.075, 0.1, 0.2])

    for case in range(400):
        height = int(rng.integers(0, 13))
        width = int(rng.integers(1, 17))
        if height == 0:
            results = {}
            _assert_equivalent(results, -3.0, 5.0, width, 0.1)
            continue

        increments = rng.choice(
            increments_to_sample,
            size=height - 1,
            replace=True,
        )
        ki_values = np.r_[0.0, np.cumsum(increments)]
        density = float(rng.choice([0.2, 0.45, 0.7, 0.9]))
        matrix = rng.random((height, width)) < density
        results, kp_min, kp_max, resolution = _matrix_results(
            matrix,
            ki_values,
            kp_min=-3.0,
            kp_max=5.0,
        )

        gaps = np.diff(ki_values)
        if len(gaps) and case % 3 == 0:
            max_gap = float(gaps[int(rng.integers(0, len(gaps)))])
        else:
            max_gap = float(rng.choice(increments_to_sample))

        _assert_equivalent(
            results,
            kp_min,
            kp_max,
            resolution,
            max_gap,
        )

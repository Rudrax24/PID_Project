import pytest

from fair_benchmark import (
    FairConfig,
    _EvaluationCounts,
    _refine_transition,
    lattice_points_for_ranges,
    run_search,
)


def test_ranges_use_the_shared_lattice_and_include_endpoints():
    config = FairConfig(kp_min=-10, kp_max=10, coarse_points=40)

    points = lattice_points_for_ranges(
        config.kp_min,
        config.kp_max,
        config.coarse_points,
        [(-2.2, 1.7)],
    )
    lattice_in_range = [
        point
        for point in (
            config.kp_min + i * config.kp_spacing
            for i in range(config.coarse_points)
        )
        if -2.2 <= point <= 1.7
    ]

    assert points[0] == -2.2
    assert points[-1] == 1.7
    assert set(lattice_in_range).issubset(set(points))
    assert all(
        abs((right - left) - config.kp_spacing) < 1e-12
        for left, right in zip(lattice_in_range[:-1], lattice_in_range[1:])
    )


def test_binary_refinement_uses_one_absolute_tolerance():
    config = FairConfig(kp_min=-2, kp_max=2, coarse_points=5, binary_iterations=4)
    counts = _EvaluationCounts()

    boundary = _refine_transition(
        low=0,
        high=1,
        low_stable=False,
        Ki=0,
        plant=None,
        Kd=0,
        stability_fn=lambda Kp, _Ki, _Kd, _plant: Kp >= 0.37,
        counts=counts,
        tolerance=config.binary_tolerance,
    )

    assert abs(boundary - 0.37) <= config.binary_tolerance
    assert counts.values["binary_refinement_evaluations"] > 0


def test_row_accounting_includes_empty_terminating_rows():
    config = FairConfig(
        kp_min=-2,
        kp_max=2,
        ki_min=-3,
        ki_max=3,
        initial_step=1,
        refined_step=1,
        coarse_points=5,
        binary_iterations=4,
        plot_resolution=10,
    )

    def classifier(Kp, Ki, _Kd, _plant):
        return -1 <= Kp <= 1 and Ki <= 1

    run = run_search(
        plant=None,
        stability_fn=classifier,
        rectangle_fn=lambda **_kwargs: None,
        strategy="exhaustive",
        config=config,
    )

    assert run["summary"]["ki_rows_processed"] == 6
    assert run["summary"]["empty_rows"] == 1
    assert run["summary"]["terminating_rows"] == 1
    assert len(run["rows"]) == 6
    assert any(row["status"] == "terminating" for row in run["rows"])

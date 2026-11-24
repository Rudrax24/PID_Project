import pytest

from stability_search import (
    SearchStats,
    _boundary_only_row,
    search_direction,
)


def interval_classifier(intervals_by_ki):
    def is_stable(Kp, Ki, Kd, plant):
        return any(left <= Kp <= right for left, right in intervals_by_ki(Ki))

    return is_stable


@pytest.mark.parametrize(
    ("new_interval", "expected"),
    [
        ((-2.2, 2.2), (-2.2, 2.2)),
        ((-1.8, 1.8), (-1.8, 1.8)),
    ],
)
def test_boundary_only_reconstructs_expansion_and_contraction(new_interval, expected):
    classifier = interval_classifier(lambda _ki: [new_interval])
    stats = SearchStats(strategy="boundary_only")

    intervals = _boundary_only_row(
        Ki=1,
        previous_intervals=[(-2, 2)],
        plant=None,
        Kd=0,
        kp_min=-5,
        kp_max=5,
        search_margin=0.5,
        coarse_steps=41,
        binary_iterations=16,
        stability_fn=classifier,
        stats=stats,
        sentinel_count=3,
        sentinel_positions=None,
    )

    assert intervals[0] == pytest.approx(expected, abs=1e-4)
    assert stats.full_row_fallback_scans == 0
    assert stats.sentinel_evaluations >= 3


def test_interior_sentinel_failure_falls_back_and_finds_split_intervals():
    classifier = interval_classifier(
        lambda _ki: [(-2, -0.02), (0.02, 2)]
    )
    stats = SearchStats(strategy="boundary_only")

    intervals = _boundary_only_row(
        Ki=1,
        previous_intervals=[(-2, 2)],
        plant=None,
        Kd=0,
        kp_min=-5,
        kp_max=5,
        search_margin=0.5,
        coarse_steps=101,
        binary_iterations=16,
        stability_fn=classifier,
        stats=stats,
        sentinel_count=3,
        sentinel_positions=None,
    )

    assert len(intervals) == 2
    assert stats.full_row_fallback_scans == 1


def test_boundary_only_reconstructs_multiple_disjoint_intervals():
    classifier = interval_classifier(
        lambda _ki: [(-4.2, -1.8), (1.8, 4.2)]
    )
    stats = SearchStats(strategy="boundary_only")

    intervals = _boundary_only_row(
        Ki=1,
        previous_intervals=[(-4, -2), (2, 4)],
        plant=None,
        Kd=0,
        kp_min=-6,
        kp_max=6,
        search_margin=0.5,
        coarse_steps=61,
        binary_iterations=16,
        stability_fn=classifier,
        stats=stats,
        sentinel_count=3,
        sentinel_positions=None,
    )

    assert len(intervals) == 2
    assert intervals[0] == pytest.approx((-4.2, -1.8), abs=1e-4)
    assert intervals[1] == pytest.approx((1.8, 4.2), abs=1e-4)
    assert stats.full_row_fallback_scans == 0


def test_boundary_only_respects_global_kp_limits():
    classifier = interval_classifier(lambda _ki: [(-5, -2.8)])
    stats = SearchStats(strategy="boundary_only")

    intervals = _boundary_only_row(
        Ki=1,
        previous_intervals=[(-5, -3)],
        plant=None,
        Kd=0,
        kp_min=-5,
        kp_max=5,
        search_margin=0.5,
        coarse_steps=41,
        binary_iterations=16,
        stability_fn=classifier,
        stats=stats,
        sentinel_count=3,
        sentinel_positions=None,
    )

    assert intervals[0] == pytest.approx((-5, -2.8), abs=1e-4)
    assert stats.full_row_fallback_scans == 0


def test_disappearing_interval_falls_back_to_empty_full_row():
    classifier = interval_classifier(lambda _ki: [])
    stats = SearchStats(strategy="boundary_only")

    intervals = _boundary_only_row(
        Ki=1,
        previous_intervals=[(-2, 2)],
        plant=None,
        Kd=0,
        kp_min=-5,
        kp_max=5,
        search_margin=0.5,
        coarse_steps=41,
        binary_iterations=16,
        stability_fn=classifier,
        stats=stats,
        sentinel_count=3,
        sentinel_positions=None,
    )

    assert intervals == []
    assert stats.full_row_fallback_scans == 1


def test_sparse_sentinels_can_miss_a_narrow_interior_instability():
    classifier = interval_classifier(
        lambda _ki: [(-2, -0.05), (0.05, 2)]
    )
    stats = SearchStats(strategy="boundary_only")

    intervals = _boundary_only_row(
        Ki=1,
        previous_intervals=[(-2, 2)],
        plant=None,
        Kd=0,
        kp_min=-5,
        kp_max=5,
        search_margin=0.5,
        coarse_steps=41,
        binary_iterations=16,
        stability_fn=classifier,
        stats=stats,
        sentinel_count=2,
        sentinel_positions=[0.25, 0.75],
    )

    assert intervals[0] == pytest.approx((-2, 2), abs=1e-4)
    assert stats.full_row_fallback_scans == 0
    assert not classifier(0, 1, 0, None)


def test_periodic_full_row_validation_is_counted():
    classifier = interval_classifier(lambda _ki: [(-1, 1)])
    stats = SearchStats(strategy="boundary_only")

    results = search_direction(
        seed_ki=0,
        direction=1,
        plant=None,
        Kd=0,
        ki_min=0,
        ki_max=2,
        kp_min=-5,
        kp_max=5,
        stability_fn=classifier,
        initial_step=1,
        refined_step=1,
        search_margin=0.5,
        coarse_steps=41,
        binary_iterations=16,
        strategy="boundary_only",
        stats=stats,
        periodic_validation_interval=1,
    )

    assert set(results) == {1, 2}
    assert stats.periodic_full_row_validations == 1
    assert stats.ki_rows_processed == 2
    assert stats.total_is_stable_calls == (
        stats.coarse_kp_evaluations
        + stats.binary_search_evaluations
        + stats.sentinel_evaluations
    )


def test_periodic_validation_falls_back_when_it_detects_an_interior_split():
    classifier = interval_classifier(
        lambda Ki: [(-2, 2)] if Ki == 1 else [(-2, -0.05), (0.05, 2)]
    )
    stats = SearchStats(strategy="boundary_only")

    results = search_direction(
        seed_ki=0,
        direction=1,
        plant=None,
        Kd=0,
        ki_min=0,
        ki_max=2,
        kp_min=-5,
        kp_max=5,
        stability_fn=classifier,
        initial_step=1,
        refined_step=1,
        search_margin=0.5,
        coarse_steps=101,
        binary_iterations=16,
        strategy="boundary_only",
        stats=stats,
        sentinel_positions=[0.25, 0.75],
        periodic_validation_interval=1,
    )

    assert len(results[2]) == 2
    assert stats.periodic_full_row_validations == 1
    assert stats.full_row_fallback_scans == 1


def test_search_direction_processes_negative_ki_rows():
    classifier = interval_classifier(lambda _ki: [(-1, 1)])

    results = search_direction(
        seed_ki=0,
        direction=-1,
        plant=None,
        Kd=0,
        ki_min=-2,
        ki_max=0,
        kp_min=-5,
        kp_max=5,
        stability_fn=classifier,
        initial_step=1,
        refined_step=1,
        coarse_steps=21,
        binary_iterations=8,
        strategy="exhaustive",
    )

    assert set(results) == {-1, -2}

"""Resolution-matched benchmark harness; does not change the search module."""

from dataclasses import dataclass
from math import ceil, floor
from time import perf_counter

import numpy as np


STRATEGIES = ("exhaustive", "expanded_interval", "boundary_only")


@dataclass(frozen=True)
class FairConfig:
    kp_min: float = -10.0
    kp_max: float = 10.0
    ki_min: float = -10.0
    ki_max: float = 10.0
    seed_ki: float = 0.0
    initial_step: float = 0.1
    refined_step: float = 0.1
    search_margin: float = 0.5
    coarse_points: int = 40
    binary_iterations: int = 20
    plot_resolution: int = 300
    sentinel_count: int = 3
    sentinel_positions: tuple | None = None
    periodic_validation_interval: int = 5

    @property
    def kp_spacing(self):
        return (self.kp_max - self.kp_min) / (self.coarse_points - 1)

    @property
    def binary_tolerance(self):
        return self.kp_spacing / (2 ** self.binary_iterations)


class _EvaluationCounts:
    def __init__(self):
        self.values = {
            "total_classifier_evaluations": 0,
            "coarse_evaluations": 0,
            "binary_refinement_evaluations": 0,
            "sentinel_evaluations": 0,
            "full_row_fallback_evaluations": 0,
            "periodic_full_row_evaluations": 0,
            "full_row_fallback_scans": 0,
            "periodic_full_row_validations": 0,
        }

    def evaluate(self, stability_fn, Kp, Ki, Kd, plant, category, scan_kind=None):
        self.values["total_classifier_evaluations"] += 1
        if category == "coarse":
            self.values["coarse_evaluations"] += 1
        elif category == "binary":
            self.values["binary_refinement_evaluations"] += 1
        elif category == "sentinel":
            self.values["sentinel_evaluations"] += 1
        else:
            raise ValueError(f"Unknown evaluation category: {category}")

        if scan_kind == "fallback":
            self.values["full_row_fallback_evaluations"] += 1
        elif scan_kind == "periodic":
            self.values["periodic_full_row_evaluations"] += 1

        return bool(stability_fn(Kp, Ki, Kd, plant))


def merge_ranges(ranges):
    ordered = sorted((float(left), float(right)) for left, right in ranges if right >= left)
    if not ordered:
        return []

    merged = [list(ordered[0])]
    for left, right in ordered[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [tuple(interval) for interval in merged]


def lattice_points_for_ranges(kp_min, kp_max, coarse_points, ranges):
    """Return shared-lattice points in ranges plus each permitted range endpoint."""
    spacing = (kp_max - kp_min) / (coarse_points - 1)
    points = []
    for left, right in merge_ranges(ranges):
        if left < kp_min or right > kp_max:
            raise ValueError("Search ranges must lie within the global Kp limits.")
        points.extend((left, right))
        first = max(0, ceil((left - kp_min) / spacing - 1e-12))
        last = min(coarse_points - 1, floor((right - kp_min) / spacing + 1e-12))
        points.extend(kp_min + index * spacing for index in range(first, last + 1))

    points.sort()
    unique = []
    tolerance = max(1e-12, abs(kp_max - kp_min) * 1e-12)
    for point in points:
        if not unique or point - unique[-1] > tolerance:
            unique.append(float(point))
    return unique


def _refine_transition(
    low,
    high,
    low_stable,
    Ki,
    plant,
    Kd,
    stability_fn,
    counts,
    tolerance,
    scan_kind=None,
):
    while high - low > tolerance:
        midpoint = (low + high) / 2
        midpoint_stable = counts.evaluate(
            stability_fn, midpoint, Ki, Kd, plant, "binary", scan_kind
        )
        if midpoint_stable == low_stable:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _intervals_from_samples(
    points,
    stable_samples,
    Ki,
    plant,
    Kd,
    stability_fn,
    counts,
    tolerance,
    scan_kind=None,
):
    intervals = []
    index = 0
    while index < len(points):
        if not stable_samples[index]:
            index += 1
            continue

        start = index
        while index + 1 < len(points) and stable_samples[index + 1]:
            index += 1
        end = index

        if start == 0:
            left = points[0]
        else:
            left = _refine_transition(
                points[start - 1],
                points[start],
                stable_samples[start - 1],
                Ki,
                plant,
                Kd,
                stability_fn,
                counts,
                tolerance,
                scan_kind,
            )

        if end == len(points) - 1:
            right = points[-1]
        else:
            right = _refine_transition(
                points[end],
                points[end + 1],
                stable_samples[end],
                Ki,
                plant,
                Kd,
                stability_fn,
                counts,
                tolerance,
                scan_kind,
            )

        intervals.append((left, right))
        index += 1

    return merge_ranges(intervals)


def _scan_ranges(
    Ki,
    ranges,
    plant,
    Kd,
    stability_fn,
    counts,
    config,
    scan_kind=None,
):
    intervals = []
    for left, right in merge_ranges(ranges):
        points = lattice_points_for_ranges(
            config.kp_min,
            config.kp_max,
            config.coarse_points,
            [(left, right)],
        )
        samples = [
            counts.evaluate(
                stability_fn, point, Ki, Kd, plant, "coarse", scan_kind
            )
            for point in points
        ]
        intervals.extend(
            _intervals_from_samples(
                points,
                samples,
                Ki,
                plant,
                Kd,
                stability_fn,
                counts,
                config.binary_tolerance,
                scan_kind,
            )
        )
    return merge_ranges(intervals)


def _full_row_scan(Ki, plant, Kd, stability_fn, counts, config, scan_kind=None):
    return _scan_ranges(
        Ki,
        [(config.kp_min, config.kp_max)],
        plant,
        Kd,
        stability_fn,
        counts,
        config,
        scan_kind,
    )


def _boundary_row(
    Ki,
    previous_intervals,
    plant,
    Kd,
    stability_fn,
    counts,
    config,
):
    def fallback(reason):
        counts.values["full_row_fallback_scans"] += 1
        intervals = _full_row_scan(
            Ki, plant, Kd, stability_fn, counts, config, scan_kind="fallback"
        )
        return intervals, reason, True

    if not previous_intervals:
        return fallback("missing_previous_intervals")

    if config.sentinel_positions is None:
        if config.sentinel_count < 1:
            raise ValueError("sentinel_count must be at least one.")
        positions = [
            index / (config.sentinel_count + 1)
            for index in range(1, config.sentinel_count + 1)
        ]
    else:
        positions = sorted(set(float(value) for value in config.sentinel_positions))
        if not positions or any(value <= 0 or value >= 1 for value in positions):
            raise ValueError("Sentinel positions must be fractions between 0 and 1.")

    windows = []
    outer_edges = []
    sentinel_points = set()
    for left, right in previous_intervals:
        if not (config.kp_min <= left <= right <= config.kp_max):
            return fallback("invalid_previous_interval")

        width = right - left
        sentinel_points.update(left + position * width for position in positions)
        left_anchor = min(right, left + config.search_margin)
        right_anchor = max(left, right - config.search_margin)
        sentinel_points.update((left_anchor, right_anchor))

        left_window = (
            max(config.kp_min, left - config.search_margin),
            min(config.kp_max, left + config.search_margin),
        )
        right_window = (
            max(config.kp_min, right - config.search_margin),
            min(config.kp_max, right + config.search_margin),
        )
        windows.extend((left_window, right_window))
        if left_window[0] > config.kp_min + 1e-10:
            outer_edges.append(left_window[0])
        if right_window[1] < config.kp_max - 1e-10:
            outer_edges.append(right_window[1])

    for point in sorted(sentinel_points):
        if not counts.evaluate(
            stability_fn, point, Ki, Kd, plant, "sentinel"
        ):
            return fallback("unstable_sentinel")

    merged_windows = merge_ranges(windows)
    observed = _scan_ranges(
        Ki,
        merged_windows,
        plant,
        Kd,
        stability_fn,
        counts,
        config,
    )
    tolerance = max(1e-10, (config.kp_max - config.kp_min) * 1e-10)

    for edge in outer_edges:
        if any(left - tolerance <= edge <= right + tolerance for left, right in observed):
            return fallback("stable_at_window_edge")

    for observed_left, observed_right in observed:
        if not any(
            observed_right >= old_left - tolerance
            and observed_left <= old_right + tolerance
            for old_left, old_right in previous_intervals
        ):
            return fallback("disconnected_stable_window")

    reconstructed = []
    for old_left, old_right in previous_intervals:
        left_anchor = min(old_right, old_left + config.search_margin)
        right_anchor = max(old_left, old_right - config.search_margin)
        left_runs = [
            interval for interval in observed
            if interval[0] - tolerance <= left_anchor <= interval[1] + tolerance
        ]
        right_runs = [
            interval for interval in observed
            if interval[0] - tolerance <= right_anchor <= interval[1] + tolerance
        ]
        if not left_runs or not right_runs:
            return fallback("missing_boundary_anchor")

        left_run = left_runs[0]
        right_run = right_runs[0]
        if left_run[0] > right_run[1] + tolerance:
            return fallback("inconsistent_boundary_order")

        if left_run != right_run and any(
            window_left <= left_anchor + tolerance
            and window_right >= right_anchor - tolerance
            for window_left, window_right in merged_windows
        ):
            return fallback("observed_split")

        reconstructed.append((left_run[0], right_run[1]))

    reconstructed = merge_ranges(reconstructed)
    for observed_left, observed_right in observed:
        if not any(
            candidate_left <= observed_left + tolerance
            and candidate_right >= observed_right - tolerance
            for candidate_left, candidate_right in reconstructed
        ):
            return fallback("unexplained_window_interval")

    return reconstructed, "", False


def _intervals_match(expected, actual, tolerance):
    if len(expected) != len(actual):
        return False
    return all(
        abs(expected_left - actual_left) <= tolerance
        and abs(expected_right - actual_right) <= tolerance
        for (expected_left, expected_right), (actual_left, actual_right)
        in zip(expected, actual)
    )


def run_search(
    plant,
    stability_fn,
    rectangle_fn,
    strategy,
    config,
    sentinel_count=None,
    sentinel_positions=None,
):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if config.coarse_points < 2:
        raise ValueError("coarse_points must be at least two.")
    if config.periodic_validation_interval < 0:
        raise ValueError("periodic_validation_interval cannot be negative.")

    if sentinel_count is not None or sentinel_positions is not None:
        config = FairConfig(
            **{
                **config.__dict__,
                "sentinel_count": (
                    config.sentinel_count if sentinel_count is None else sentinel_count
                ),
                "sentinel_positions": (
                    config.sentinel_positions
                    if sentinel_positions is None
                    else tuple(sentinel_positions)
                ),
            }
        )

    counts = _EvaluationCounts()
    attempted_rows = []
    retained_results = {}
    search_started = perf_counter()

    seed_intervals = _full_row_scan(
        config.seed_ki, plant, 0, stability_fn, counts, config
    )
    seed_key = round(config.seed_ki, 8)
    retained_results[seed_key] = seed_intervals
    attempted_rows.append(
        {
            "Ki": seed_key,
            "direction": 0,
            "row_kind": "seed",
            "status": "stable" if seed_intervals else "unstable",
            "interval_count": len(seed_intervals),
            "intervals": seed_intervals,
            "full_row_fallback": False,
            "fallback_reason": "",
            "periodic_validated": False,
            "retained_in_original_results": True,
        }
    )

    for direction in (1, -1):
        Ki = config.seed_ki
        found_stability = False
        previous_intervals = []
        rows_since_stable = 0

        while True:
            step = config.refined_step if found_stability else config.initial_step
            Ki = Ki + direction * step
            if Ki < config.ki_min - 1e-9 or Ki > config.ki_max + 1e-9:
                break
            row_ki = round(Ki, 8)

            found_before_row = found_stability
            if found_before_row:
                rows_since_stable += 1

            fallback_used = False
            fallback_reason = ""
            periodic_validated = False

            if strategy == "exhaustive" or not found_stability:
                intervals = _full_row_scan(
                    Ki, plant, 0, stability_fn, counts, config
                )
            elif strategy == "expanded_interval":
                ranges = merge_ranges(
                    (
                        max(config.kp_min, left - config.search_margin),
                        min(config.kp_max, right + config.search_margin),
                    )
                    for left, right in previous_intervals
                )
                intervals = _scan_ranges(
                    Ki, ranges, plant, 0, stability_fn, counts, config
                )
            else:
                intervals, fallback_reason, fallback_used = _boundary_row(
                    Ki,
                    previous_intervals,
                    plant,
                    0,
                    stability_fn,
                    counts,
                    config,
                )

                if (
                    found_before_row
                    and config.periodic_validation_interval
                    and rows_since_stable > 0
                    and rows_since_stable % config.periodic_validation_interval == 0
                ):
                    periodic_validated = True
                    counts.values["periodic_full_row_validations"] += 1
                    calls_before = counts.values["total_classifier_evaluations"]
                    reference_intervals = _full_row_scan(
                        Ki,
                        plant,
                        0,
                        stability_fn,
                        counts,
                        config,
                        scan_kind="periodic",
                    )
                    validation_calls = (
                        counts.values["total_classifier_evaluations"] - calls_before
                    )
                    validation_tolerance = max(
                        1e-8,
                        (2 * config.search_margin) / (config.coarse_points - 1),
                    )
                    if not _intervals_match(
                        intervals, reference_intervals, validation_tolerance
                    ):
                        intervals = reference_intervals
                        fallback_used = True
                        fallback_reason = "periodic_validation_mismatch"
                        counts.values["full_row_fallback_scans"] += 1
                        counts.values["full_row_fallback_evaluations"] += validation_calls

            if not intervals:
                status = "terminating" if found_stability else "unstable"
            else:
                status = "stable"

            attempted_rows.append(
                {
                    "Ki": row_ki,
                    "direction": direction,
                    "row_kind": "search",
                    "status": status,
                    "interval_count": len(intervals),
                    "intervals": intervals,
                    "full_row_fallback": fallback_used,
                    "fallback_reason": fallback_reason,
                    "periodic_validated": periodic_validated,
                    "retained_in_original_results": bool(intervals),
                }
            )

            if not intervals:
                if found_stability:
                    break
                continue

            if not found_stability:
                rows_since_stable = 0
            found_stability = True
            previous_intervals = intervals
            retained_results[row_ki] = intervals

    search_seconds = perf_counter() - search_started

    retained_results = dict(sorted(retained_results.items()))
    rectangle_started = perf_counter()
    rectangle = rectangle_fn(
        results=retained_results,
        kp_min=config.kp_min,
        kp_max=config.kp_max,
        plot_resolution=config.plot_resolution,
        max_ki_gap=config.refined_step * 1.5,
    )
    rectangle_seconds = perf_counter() - rectangle_started

    stable_interval_count = sum(
        row["interval_count"] for row in attempted_rows
    )
    summary = {
        "strategy": strategy,
        "sentinel_count": config.sentinel_count,
        "kp_spacing": config.kp_spacing,
        "binary_tolerance": config.binary_tolerance,
        "ki_rows_processed": len(attempted_rows),
        "retained_rows": sum(row["retained_in_original_results"] for row in attempted_rows),
        "empty_rows": sum(row["interval_count"] == 0 for row in attempted_rows),
        "terminating_rows": sum(row["status"] == "terminating" for row in attempted_rows),
        "fallback_rows": sum(row["full_row_fallback"] for row in attempted_rows),
        "periodic_validated_rows": sum(row["periodic_validated"] for row in attempted_rows),
        "stable_intervals_detected": stable_interval_count,
        "search_time_seconds": search_seconds,
        "rectangle_time_seconds": rectangle_seconds,
        "total_execution_time_seconds": search_seconds + rectangle_seconds,
        **counts.values,
    }

    return {
        "summary": summary,
        "rows": attempted_rows,
        "retained_results": retained_results,
        "rectangle": rectangle,
    }


def dense_reference(
    Ki,
    plant,
    stability_fn,
    config,
    subdivisions=10,
):
    dense_points = (config.coarse_points - 1) * subdivisions + 1
    points = np.linspace(config.kp_min, config.kp_max, dense_points)
    stable_samples = [
        bool(stability_fn(Kp, Ki, 0, plant))
        for Kp in points
    ]

    class _DenseCounter:
        def __init__(self):
            self.calls = 0

        def evaluate(self, stability_fn, Kp, Ki, Kd, plant, category, scan_kind=None):
            self.calls += 1
            return bool(stability_fn(Kp, Ki, Kd, plant))

    counter = _DenseCounter()

    def refine(low, high, low_stable):
        return _refine_transition(
            low,
            high,
            low_stable,
            Ki,
            plant,
            0,
            stability_fn,
            counter,
            config.binary_tolerance,
        )

    intervals = []
    index = 0
    while index < len(points):
        if not stable_samples[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(points) and stable_samples[index + 1]:
            index += 1
        end = index
        left = points[0] if start == 0 else refine(
            points[start - 1], points[start], stable_samples[start - 1]
        )
        right = points[-1] if end == len(points) - 1 else refine(
            points[end], points[end + 1], stable_samples[end]
        )
        intervals.append((float(left), float(right)))
        index += 1

    return {
        "intervals": merge_ranges(intervals),
        "grid_points": points,
        "stable_samples": np.asarray(stable_samples, dtype=bool),
        "classifier_evaluations": len(points) + counter.calls,
        "spacing": (config.kp_max - config.kp_min) / (dense_points - 1),
    }


def compare_to_dense(adaptive_intervals, reference):
    points = reference["grid_points"]
    reference_mask = reference["stable_samples"]
    adaptive_mask = np.zeros(len(points), dtype=bool)
    for left, right in adaptive_intervals:
        adaptive_mask |= (points >= left) & (points <= right)

    paired = min(len(adaptive_intervals), len(reference["intervals"]))
    left_errors = [
        abs(adaptive_intervals[index][0] - reference["intervals"][index][0])
        for index in range(paired)
    ]
    right_errors = [
        abs(adaptive_intervals[index][1] - reference["intervals"][index][1])
        for index in range(paired)
    ]

    missed_islands = 0
    reference_intervals = reference["intervals"]
    for left_interval, right_interval in zip(
        reference_intervals[:-1], reference_intervals[1:]
    ):
        gap_midpoint = (left_interval[1] + right_interval[0]) / 2
        if any(left <= gap_midpoint <= right for left, right in adaptive_intervals):
            missed_islands += 1

    missed_regions = sum(
        not any(
            candidate_right >= reference_left
            and candidate_left <= reference_right
            for candidate_left, candidate_right in adaptive_intervals
        )
        for reference_left, reference_right in reference_intervals
    )
    false_regions = sum(
        not any(
            reference_right >= candidate_left
            and reference_left <= candidate_right
            for reference_left, reference_right in reference_intervals
        )
        for candidate_left, candidate_right in adaptive_intervals
    )

    total_points = max(1, len(points))
    return {
        "dense_grid_points": len(points),
        "dense_reference_intervals": len(reference_intervals),
        "adaptive_intervals": len(adaptive_intervals),
        "missed_stable_regions": missed_regions,
        "false_stable_regions": false_regions,
        "missed_stable_grid_points": int(np.count_nonzero(reference_mask & ~adaptive_mask)),
        "false_stable_grid_points": int(np.count_nonzero(adaptive_mask & ~reference_mask)),
        "grid_agreement": float(np.count_nonzero(reference_mask == adaptive_mask) / total_points),
        "left_boundary_error_mean": float(np.mean(left_errors)) if left_errors else np.nan,
        "left_boundary_error_max": float(np.max(left_errors)) if left_errors else np.nan,
        "right_boundary_error_mean": float(np.mean(right_errors)) if right_errors else np.nan,
        "right_boundary_error_max": float(np.max(right_errors)) if right_errors else np.nan,
        "narrow_unstable_islands_missed": missed_islands,
    }

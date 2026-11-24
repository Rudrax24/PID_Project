"""Adaptive Kp/Ki stability-search strategies."""

from dataclasses import dataclass, asdict
from math import ceil

import numpy as np


STRATEGIES = ("exhaustive", "expanded_interval", "boundary_only")


@dataclass
class SearchStats:
    strategy: str
    total_is_stable_calls: int = 0
    coarse_kp_evaluations: int = 0
    binary_search_evaluations: int = 0
    sentinel_evaluations: int = 0
    ki_rows_processed: int = 0
    full_row_fallback_scans: int = 0
    periodic_full_row_validations: int = 0
    stable_intervals_detected: int = 0
    search_time_seconds: float = 0.0
    rectangle_time_seconds: float = 0.0

    def evaluate(self, category, stability_fn, Kp, Ki, Kd, plant):
        if category == "coarse":
            self.coarse_kp_evaluations += 1
        elif category == "binary":
            self.binary_search_evaluations += 1
        elif category == "sentinel":
            self.sentinel_evaluations += 1
        else:
            raise ValueError(f"Unknown evaluation category: {category}")

        self.total_is_stable_calls += 1
        return bool(stability_fn(Kp, Ki, Kd, plant))

    def as_dict(self):
        values = asdict(self)
        values["total_execution_time_seconds"] = (
            self.search_time_seconds + self.rectangle_time_seconds
        )
        return values


def _evaluate(stats, category, stability_fn, Kp, Ki, Kd, plant):
    if stats is None:
        return bool(stability_fn(Kp, Ki, Kd, plant))
    return stats.evaluate(category, stability_fn, Kp, Ki, Kd, plant)


def merge_ranges(ranges):
    ranges = list(ranges)
    if not ranges:
        return []

    sorted_ranges = sorted((float(left), float(right)) for left, right in ranges)
    merged = [list(sorted_ranges[0])]

    for left, right in sorted_ranges[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])

    return [tuple(interval) for interval in merged]


def refine_boundary(
    low,
    high,
    Ki,
    plant,
    Kd,
    stability_fn,
    iterations=20,
    stats=None,
):
    low_stable = _evaluate(stats, "binary", stability_fn, low, Ki, Kd, plant)

    for _ in range(iterations):
        mid = (low + high) / 2
        mid_stable = _evaluate(stats, "binary", stability_fn, mid, Ki, Kd, plant)

        if mid_stable == low_stable:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def scan_ki_row(
    Ki,
    plant,
    Kd,
    kp_ranges,
    stability_fn,
    coarse_steps=80,
    binary_iterations=20,
    stats=None,
):
    stable_intervals = []

    for kp_start, kp_end in kp_ranges:
        if kp_end <= kp_start:
            continue

        kp_values = np.linspace(kp_start, kp_end, coarse_steps)
        stable_samples = [
            _evaluate(stats, "coarse", stability_fn, kp, Ki, Kd, plant)
            for kp in kp_values
        ]

        i = 0
        while i < len(stable_samples):
            if not stable_samples[i]:
                i += 1
                continue

            start_idx = i
            while i + 1 < len(stable_samples) and stable_samples[i + 1]:
                i += 1
            end_idx = i

            if start_idx == 0:
                left_boundary = kp_values[0]
            else:
                left_boundary = refine_boundary(
                    kp_values[start_idx - 1],
                    kp_values[start_idx],
                    Ki,
                    plant,
                    Kd,
                    stability_fn,
                    binary_iterations,
                    stats,
                )

            if end_idx == len(stable_samples) - 1:
                right_boundary = kp_values[-1]
            else:
                right_boundary = refine_boundary(
                    kp_values[end_idx],
                    kp_values[end_idx + 1],
                    Ki,
                    plant,
                    Kd,
                    stability_fn,
                    binary_iterations,
                    stats,
                )

            stable_intervals.append((left_boundary, right_boundary))
            i += 1

    return merge_ranges(stable_intervals)


def _full_row_scan(
    Ki,
    plant,
    Kd,
    kp_min,
    kp_max,
    stability_fn,
    coarse_steps,
    binary_iterations,
    stats,
    fallback=False,
):
    if stats is not None and fallback:
        stats.full_row_fallback_scans += 1

    return scan_ki_row(
        Ki,
        plant,
        Kd,
        [(kp_min, kp_max)],
        stability_fn,
        coarse_steps,
        binary_iterations,
        stats,
    )


def _boundary_only_row(
    Ki,
    previous_intervals,
    plant,
    Kd,
    kp_min,
    kp_max,
    search_margin,
    coarse_steps,
    binary_iterations,
    stability_fn,
    stats,
    sentinel_count,
    sentinel_positions,
):
    if not previous_intervals:
        return _full_row_scan(
            Ki, plant, Kd, kp_min, kp_max, stability_fn,
            coarse_steps, binary_iterations, stats, fallback=True
        )

    if sentinel_positions is None:
        if sentinel_count < 1:
            raise ValueError("sentinel_count must be at least 1.")
        positions = [i / (sentinel_count + 1) for i in range(1, sentinel_count + 1)]
    else:
        positions = sorted(set(float(position) for position in sentinel_positions))
        if not positions or any(position <= 0 or position >= 1 for position in positions):
            raise ValueError("sentinel_positions must contain fractions strictly between 0 and 1.")

    windows = []
    outer_edges = []
    sentinels = set()

    for left, right in previous_intervals:
        if not (kp_min <= left <= right <= kp_max):
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

        width = right - left
        sentinels.update(left + position * width for position in positions)
        left_anchor = min(right, left + search_margin)
        right_anchor = max(left, right - search_margin)
        sentinels.update((left_anchor, right_anchor))

        left_window = (
            max(kp_min, left - search_margin),
            min(kp_max, left + search_margin),
        )
        right_window = (
            max(kp_min, right - search_margin),
            min(kp_max, right + search_margin),
        )
        windows.extend((left_window, right_window))
        if left_window[0] > kp_min + 1e-10:
            outer_edges.append(left_window[0])
        if right_window[1] < kp_max - 1e-10:
            outer_edges.append(right_window[1])

    for kp in sorted(sentinels):
        if not _evaluate(stats, "sentinel", stability_fn, kp, Ki, Kd, plant):
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

    merged_windows = merge_ranges(windows)
    if not merged_windows:
        return _full_row_scan(
            Ki, plant, Kd, kp_min, kp_max, stability_fn,
            coarse_steps, binary_iterations, stats, fallback=True
        )

    full_row_spacing = (kp_max - kp_min) / max(coarse_steps - 1, 1)
    observed_intervals = []
    for window_left, window_right in merged_windows:
        window_steps = min(
            coarse_steps,
            max(2, ceil((window_right - window_left) / full_row_spacing) + 1),
        )
        observed_intervals.extend(
            scan_ki_row(
                Ki,
                plant,
                Kd,
                [(window_left, window_right)],
                stability_fn,
                window_steps,
                binary_iterations,
                stats,
            )
        )
    observed_intervals = merge_ranges(observed_intervals)

    tolerance = max(1e-10, (kp_max - kp_min) * 1e-10)

    # A stable sample at an outer window edge means the transition may lie
    # beyond the observed neighborhood; do not assume that edge is valid.
    for edge in outer_edges:
        if any(left - tolerance <= edge <= right + tolerance for left, right in observed_intervals):
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

    # New stable runs near an old edge are ambiguous unless they connect to
    # one of the previously known intervals.
    for observed_left, observed_right in observed_intervals:
        if not any(
            observed_right >= old_left - tolerance
            and observed_left <= old_right + tolerance
            for old_left, old_right in previous_intervals
        ):
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

    reconstructed = []
    for old_left, old_right in previous_intervals:
        left_anchor = min(old_right, old_left + search_margin)
        right_anchor = max(old_left, old_right - search_margin)

        left_runs = [
            interval for interval in observed_intervals
            if interval[0] - tolerance <= left_anchor <= interval[1] + tolerance
        ]
        right_runs = [
            interval for interval in observed_intervals
            if interval[0] - tolerance <= right_anchor <= interval[1] + tolerance
        ]

        if not left_runs or not right_runs:
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

        left_run = left_runs[0]
        right_run = right_runs[0]
        if left_run[0] > right_run[1] + tolerance:
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

        if left_run != right_run and any(
            window_left <= left_anchor + tolerance
            and window_right >= right_anchor - tolerance
            for window_left, window_right in merged_windows
        ):
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

        reconstructed.append((left_run[0], right_run[1]))

    reconstructed = merge_ranges(reconstructed)

    # Every observed stable window segment must be explained by the proposed
    # reconstruction; otherwise perform the authoritative full-row scan.
    for observed_left, observed_right in observed_intervals:
        if not any(
            candidate_left <= observed_left + tolerance
            and candidate_right >= observed_right - tolerance
            for candidate_left, candidate_right in reconstructed
        ):
            return _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats, fallback=True
            )

    return reconstructed


def _intervals_match(expected, actual, tolerance):
    if len(expected) != len(actual):
        return False
    return all(
        abs(expected_left - actual_left) <= tolerance
        and abs(expected_right - actual_right) <= tolerance
        for (expected_left, expected_right), (actual_left, actual_right)
        in zip(expected, actual)
    )


def search_direction(
    seed_ki,
    direction,
    plant,
    Kd,
    ki_min,
    ki_max,
    kp_min,
    kp_max,
    stability_fn,
    initial_step=0.5,
    refined_step=0.05,
    search_margin=0.5,
    coarse_steps=80,
    binary_iterations=20,
    strategy="expanded_interval",
    stats=None,
    sentinel_count=3,
    sentinel_positions=None,
    periodic_validation_interval=5,
):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}; got {strategy!r}")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1.")
    if periodic_validation_interval < 0:
        raise ValueError("periodic_validation_interval cannot be negative.")

    results = {}
    found_stability = False
    previous_intervals = []
    Ki = seed_ki
    rows_since_stable = 0

    while True:
        step = refined_step if found_stability else initial_step
        Ki += direction * step

        if Ki < ki_min - 1e-9 or Ki > ki_max + 1e-9:
            break

        if stats is not None:
            stats.ki_rows_processed += 1
        if found_stability:
            rows_since_stable += 1

        if strategy == "exhaustive" or not found_stability:
            intervals = _full_row_scan(
                Ki, plant, Kd, kp_min, kp_max, stability_fn,
                coarse_steps, binary_iterations, stats
            )
        elif strategy == "expanded_interval":
            kp_ranges = merge_ranges(
                (
                    max(kp_min, left - search_margin),
                    min(kp_max, right + search_margin),
                )
                for left, right in previous_intervals
            )
            intervals = scan_ki_row(
                Ki, plant, Kd, kp_ranges, stability_fn,
                coarse_steps, binary_iterations, stats
            )
        else:
            intervals = _boundary_only_row(
                Ki,
                previous_intervals,
                plant,
                Kd,
                kp_min,
                kp_max,
                search_margin,
                coarse_steps,
                binary_iterations,
                stability_fn,
                stats,
                sentinel_count,
                sentinel_positions,
            )

            if (
                found_stability
                and periodic_validation_interval
                and rows_since_stable > 0
                and rows_since_stable % periodic_validation_interval == 0
            ):
                if stats is not None:
                    stats.periodic_full_row_validations += 1
                full_intervals = _full_row_scan(
                    Ki, plant, Kd, kp_min, kp_max, stability_fn,
                    coarse_steps, binary_iterations, stats
                )
                validation_tolerance = max(
                    1e-8,
                    (2 * search_margin) / max(coarse_steps - 1, 1),
                )
                if not _intervals_match(
                    intervals, full_intervals, validation_tolerance
                ):
                    if stats is not None:
                        stats.full_row_fallback_scans += 1
                    intervals = full_intervals

        if not intervals:
            if found_stability:
                print(
                    f"No stable region at Ki={Ki:.3f}. "
                    f"Stopping direction {direction:+d}."
                )
                break
            print(
                f"No stability at Ki={Ki:.3f}; "
                f"continuing with step {initial_step}."
            )
            continue

        if not found_stability:
            rows_since_stable = 0
            print(f"First stable region found at Ki={Ki:.3f}")
        found_stability = True
        previous_intervals = intervals
        results[round(Ki, 8)] = intervals

    return results

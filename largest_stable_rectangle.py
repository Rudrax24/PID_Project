"""Coordinate-aware largest stable rectangle search."""

import numpy as np


def find_largest_stable_rectangle(
    results,
    kp_min,
    kp_max,
    plot_resolution=300,
    max_ki_gap=0.075,
):
    ki_values = np.array(sorted(results.keys()), dtype=float)

    if len(ki_values) < 2:
        print("Not enough Ki rows for a 2D rectangle.")
        return None

    kp_values = np.linspace(kp_min, kp_max, plot_resolution)

    stable_map = np.zeros(
        (len(ki_values), len(kp_values)),
        dtype=bool,
    )

    for i, Ki in enumerate(ki_values):
        for left, right in results[Ki]:
            mask = (kp_values >= left) & (kp_values <= right)
            stable_map[i, mask] = True

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
            heights[column] = (
                ki_values[bottom] - ki_values[start_rows[column]]
            )

        stack = []
        for column in range(len(kp_values) + 1):
            current_height = (
                heights[column]
                if column < len(kp_values)
                else -np.inf
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

    if best_rectangle is None:
        print("No stable rectangle found.")
        return None

    kp_left, kp_right = best_rectangle["Kp_range"]
    ki_low, ki_high = best_rectangle["Ki_range"]

    print("\nLargest stable rectangle")
    print("-" * 45)
    print(f"Kp in [{kp_left:.4f}, {kp_right:.4f}]")
    print(f"Ki in [{ki_low:.4f}, {ki_high:.4f}]")
    print(f"Area = {best_area:.6f}")

    kp_center, ki_center = best_rectangle["center"]

    print("\nSuggested PID parameters:")
    print(f"Kp = {kp_center:.4f}")
    print(f"Ki = {ki_center:.4f}")

    return best_rectangle

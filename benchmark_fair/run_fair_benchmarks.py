"""Run resolution-matched and validation benchmarks into this directory."""

import contextlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_benchmark import (
    FairConfig,
    compare_to_dense,
    dense_reference,
    run_search,
)


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parent
NOTEBOOK_PATH = ROOT / "stability_without_poles.ipynb"
REPEAT_COUNT = 5
DENSE_SUBDIVISIONS = 10
STRATEGIES = ("exhaustive", "expanded_interval", "boundary_only")


def load_notebook_definitions():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for cell_index in (0, 1, 2, 3):
        source = "".join(notebook["cells"][cell_index].get("source", []))
        exec(
            compile(source, f"{NOTEBOOK_PATH.name}:cell{cell_index}", "exec"),
            namespace,
        )
    return namespace


def run_quietly(plant, stability_fn, rectangle_fn, strategy, config, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_search(
            plant,
            stability_fn,
            rectangle_fn,
            strategy,
            config,
            **kwargs,
        )


def append_run(run, run_group, plant_name, repeat_number, rows, summaries):
    summary = dict(run["summary"])
    summary.update(
        {
            "run_group": run_group,
            "plant": plant_name,
            "repeat": repeat_number,
        }
    )
    summaries.append(summary)

    for row in run["rows"]:
        rows.append(
            {
                "run_group": run_group,
                "plant": plant_name,
                "strategy": run["summary"]["strategy"],
                "sentinel_count": run["summary"]["sentinel_count"],
                "repeat": repeat_number,
                "Ki": row["Ki"],
                "direction": row["direction"],
                "row_kind": row["row_kind"],
                "status": row["status"],
                "interval_count": row["interval_count"],
                "intervals": json.dumps(row["intervals"]),
                "full_row_fallback": row["full_row_fallback"],
                "fallback_reason": row["fallback_reason"],
                "periodic_validated": row["periodic_validated"],
                "retained_in_original_results": row[
                    "retained_in_original_results"
                ],
            }
        )


def get_row_intervals(run, Ki):
    row_key = round(float(Ki), 8)
    for row in run["rows"]:
        if row["Ki"] == row_key:
            return row["intervals"]
    return []


def select_dense_rows(exhaustive_run, seed_ki):
    stable_rows = sorted(
        row["Ki"]
        for row in exhaustive_run["rows"]
        if row["interval_count"] > 0 and row["row_kind"] == "search"
    )
    selected = {round(seed_ki, 8)}
    if stable_rows:
        selected.update(
            {
                stable_rows[0],
                stable_rows[len(stable_rows) // 2],
                stable_rows[-1],
            }
        )
    selected.update(
        row["Ki"]
        for row in exhaustive_run["rows"]
        if row["status"] == "terminating"
    )
    return sorted(selected)


def main():
    namespace = load_notebook_definitions()
    systems = namespace["systems"]
    stability_fn = namespace["is_stable"]
    rectangle_fn = namespace["find_largest_stable_rectangle"]

    config = FairConfig()
    plants = [
        ("G1: 6/(s+1)(s+2)(s+3)", namespace["G1"]),
        ("G5: 120/((s+1)...(s+5))", namespace["G5"]),
        ("G6: 720/((s+1)...(s+6))", namespace["G6"]),
    ]

    original_csv = ROOT / "stability_strategy_comparison.csv"
    if original_csv.exists():
        pd.read_csv(original_csv).to_csv(
            OUTPUT_DIR / "original_implementation_benchmark.csv",
            index=False,
        )

    run_summaries = []
    row_accounting = []
    representative_runs = {}

    # Five independently timed searches per strategy on the representative G1.
    for repeat in range(1, REPEAT_COUNT + 1):
        rotation = (repeat - 1) % len(STRATEGIES)
        strategy_order = STRATEGIES[rotation:] + STRATEGIES[:rotation]
        for strategy in strategy_order:
            run = run_quietly(
                plants[0][1],
                stability_fn,
                rectangle_fn,
                strategy,
                config,
            )
            append_run(
                run,
                "repeated_G1",
                plants[0][0],
                repeat,
                row_accounting,
                run_summaries,
            )
            if repeat == 1:
                representative_runs[(plants[0][0], strategy)] = run

    # One matched three-strategy run on each additional existing plant.
    for plant_name, plant in plants[1:]:
        for strategy in STRATEGIES:
            run = run_quietly(
                plant,
                stability_fn,
                rectangle_fn,
                strategy,
                config,
            )
            append_run(
                run,
                "multi_plant",
                plant_name,
                1,
                row_accounting,
                run_summaries,
            )
            representative_runs[(plant_name, strategy)] = run

    # Compare sentinel densities without changing any other experiment setting.
    sentinel_runs = {}
    for sentinel_count in (3, 5, 9):
        if sentinel_count == config.sentinel_count:
            run = representative_runs[(plants[0][0], "boundary_only")]
        else:
            run = run_quietly(
                plants[0][1],
                stability_fn,
                rectangle_fn,
                "boundary_only",
                config,
                sentinel_count=sentinel_count,
            )
            append_run(
                run,
                "sentinel_density",
                plants[0][0],
                1,
                row_accounting,
                run_summaries,
            )
        sentinel_runs[sentinel_count] = run

    repeated_frame = pd.DataFrame(
        [
            summary
            for summary in run_summaries
            if summary["run_group"] == "repeated_G1"
        ]
    )
    repeated_frame.to_csv(OUTPUT_DIR / "repeated_run_timings.csv", index=False)
    row_frame = pd.DataFrame(row_accounting)
    row_frame.to_csv(OUTPUT_DIR / "ki_row_accounting.csv", index=False)

    timing_rows = []
    for strategy, group in repeated_frame.groupby("strategy", sort=False):
        timing_rows.append(
            {
                "strategy": strategy,
                "repeat_count": len(group),
                "mean_search_seconds": group["search_time_seconds"].mean(),
                "std_search_seconds": group["search_time_seconds"].std(ddof=1),
                "median_search_seconds": group["search_time_seconds"].median(),
                "mean_rectangle_seconds": group["rectangle_time_seconds"].mean(),
                "std_rectangle_seconds": group["rectangle_time_seconds"].std(ddof=1),
                "median_rectangle_seconds": group["rectangle_time_seconds"].median(),
                "mean_total_seconds": group["total_execution_time_seconds"].mean(),
                "std_total_seconds": group["total_execution_time_seconds"].std(ddof=1),
                "median_total_seconds": group["total_execution_time_seconds"].median(),
                "total_classifier_evaluations_per_run": group[
                    "total_classifier_evaluations"
                ].mean(),
                "coarse_evaluations_per_run": group["coarse_evaluations"].mean(),
                "binary_evaluations_per_run": group[
                    "binary_refinement_evaluations"
                ].mean(),
                "sentinel_evaluations_per_run": group["sentinel_evaluations"].mean(),
                "fallback_evaluations_per_run": group[
                    "full_row_fallback_evaluations"
                ].mean(),
                "periodic_full_row_evaluations_per_run": group[
                    "periodic_full_row_evaluations"
                ].mean(),
                "total_classifier_evaluations_all_repeats": group[
                    "total_classifier_evaluations"
                ].sum(),
            }
        )
    timing_frame = pd.DataFrame(timing_rows).set_index("strategy")
    timing_frame.to_csv(OUTPUT_DIR / "repeated_timing_summary.csv")

    # Per-plant accounting includes every attempted Ki row, not only retained rows.
    multi_frame = pd.DataFrame(
        [
            summary
            for summary in run_summaries
            if summary["run_group"] in ("repeated_G1", "multi_plant")
            and (
                summary["plant"] != plants[0][0]
                or summary["repeat"] == 1
            )
        ]
    )
    multi_frame.to_csv(OUTPUT_DIR / "multi_plant_runs.csv", index=False)

    common_rows = []
    for plant_name, _plant in plants:
        row_sets = [
            {
                row["Ki"]
                for row in representative_runs[(plant_name, strategy)]["rows"]
            }
            for strategy in STRATEGIES
        ]
        common_rows.append(
            {
                "plant": plant_name,
                "exhaustive_attempted_rows": len(row_sets[0]),
                "expanded_attempted_rows": len(row_sets[1]),
                "boundary_attempted_rows": len(row_sets[2]),
                "common_attempted_rows": len(set.intersection(*row_sets)),
            }
        )
    pd.DataFrame(common_rows).to_csv(
        OUTPUT_DIR / "common_ki_rows.csv", index=False
    )

    dense_validation_rows = []
    dense_references = {}
    for plant_name, plant in plants:
        exhaustive_run = representative_runs[(plant_name, "exhaustive")]
        selected_rows = select_dense_rows(exhaustive_run, config.seed_ki)
        for Ki in selected_rows:
            reference = dense_reference(
                Ki,
                plant,
                stability_fn,
                config,
                subdivisions=DENSE_SUBDIVISIONS,
            )
            dense_references[(plant_name, Ki)] = reference
            for strategy in STRATEGIES:
                run = representative_runs[(plant_name, strategy)]
                measured = compare_to_dense(get_row_intervals(run, Ki), reference)
                dense_validation_rows.append(
                    {
                        "plant": plant_name,
                        "Ki": Ki,
                        "strategy": strategy,
                        "dense_spacing": reference["spacing"],
                        "dense_reference_classifier_evaluations": reference[
                            "classifier_evaluations"
                        ],
                        **measured,
                    }
                )
    dense_frame = pd.DataFrame(dense_validation_rows)
    dense_frame.to_csv(OUTPUT_DIR / "dense_grid_validation.csv", index=False)

    sentinel_rows = []
    g1_selected = select_dense_rows(
        representative_runs[(plants[0][0], "exhaustive")],
        config.seed_ki,
    )
    for sentinel_count, run in sentinel_runs.items():
        metrics = run["summary"]
        dense_comparisons = []
        for Ki in g1_selected:
            reference = dense_references.get((plants[0][0], Ki))
            if reference is None:
                reference = dense_reference(
                    Ki, plants[0][1], stability_fn, config,
                    subdivisions=DENSE_SUBDIVISIONS,
                )
                dense_references[(plants[0][0], Ki)] = reference
            dense_comparisons.append(
                compare_to_dense(get_row_intervals(run, Ki), reference)
            )
        sentinel_rows.append(
            {
                "sentinel_count": sentinel_count,
                "sentinel_evaluations": metrics["sentinel_evaluations"],
                "total_classifier_evaluations": metrics[
                    "total_classifier_evaluations"
                ],
                "coarse_evaluations": metrics["coarse_evaluations"],
                "binary_refinement_evaluations": metrics[
                    "binary_refinement_evaluations"
                ],
                "full_row_fallback_scans": metrics["full_row_fallback_scans"],
                "full_row_fallback_evaluations": metrics[
                    "full_row_fallback_evaluations"
                ],
                "periodic_full_row_validations": metrics[
                    "periodic_full_row_validations"
                ],
                "search_time_seconds": metrics["search_time_seconds"],
                "rectangle_time_seconds": metrics["rectangle_time_seconds"],
                "dense_mean_grid_agreement": float(
                    np.mean([item["grid_agreement"] for item in dense_comparisons])
                ),
                "dense_missed_stable_grid_points": sum(
                    item["missed_stable_grid_points"] for item in dense_comparisons
                ),
                "dense_false_stable_grid_points": sum(
                    item["false_stable_grid_points"] for item in dense_comparisons
                ),
                "dense_missed_unstable_islands": sum(
                    item["narrow_unstable_islands_missed"]
                    for item in dense_comparisons
                ),
            }
        )
    sentinel_frame = pd.DataFrame(sentinel_rows).sort_values("sentinel_count")
    sentinel_frame.to_csv(OUTPUT_DIR / "sentinel_density_comparison.csv", index=False)

    # Reproduce the known synthetic narrow-gap limitation independently of PID plants.
    narrow_config = FairConfig(
        kp_min=-5,
        kp_max=5,
        ki_min=0,
        ki_max=2,
        seed_ki=0,
        initial_step=1,
        refined_step=1,
        search_margin=0.5,
        coarse_points=40,
        binary_iterations=20,
        plot_resolution=50,
        periodic_validation_interval=1,
    )

    def narrow_gap_classifier(Kp, Ki, _Kd, _plant):
        if Ki < 2:
            return -2 <= Kp <= 2
        return -2 <= Kp <= -0.05 or 0.05 <= Kp <= 2

    narrow_rows = []
    narrow_settings = [
        (2, (0.25, 0.75), "known_sparse_positions"),
        (3, None, "3_evenly_spaced"),
        (5, None, "5_evenly_spaced"),
        (9, None, "9_evenly_spaced"),
    ]
    for sentinel_count, positions, setting_name in narrow_settings:
        run = run_search(
            plant=None,
            stability_fn=narrow_gap_classifier,
            rectangle_fn=lambda **_kwargs: None,
            strategy="boundary_only",
            config=narrow_config,
            sentinel_count=sentinel_count,
            sentinel_positions=positions,
        )
        reference = dense_reference(
            2,
            None,
            narrow_gap_classifier,
            narrow_config,
            subdivisions=DENSE_SUBDIVISIONS,
        )
        comparison = compare_to_dense(run["retained_results"].get(2, []), reference)
        row_two = next(row for row in run["rows"] if row["Ki"] == 2)
        narrow_rows.append(
            {
                "test_case": "synthetic narrow unstable island",
                "sentinel_setting": setting_name,
                "sentinel_count": sentinel_count,
                "sentinel_positions": (
                    "default fractions" if positions is None else str(positions)
                ),
                "sentinel_evaluations": run["summary"]["sentinel_evaluations"],
                "row_2_fallback": row_two["full_row_fallback"],
                "row_2_fallback_reason": row_two["fallback_reason"],
                "periodic_full_row_validations": run["summary"][
                    "periodic_full_row_validations"
                ],
                "full_row_fallback_scans": run["summary"][
                    "full_row_fallback_scans"
                ],
                "row_2_adaptive_interval_count": len(
                    run["retained_results"].get(2, [])
                ),
                "dense_reference_interval_count": len(reference["intervals"]),
                "dense_reference_spacing": reference["spacing"],
                "dense_false_stable_grid_points": comparison[
                    "false_stable_grid_points"
                ],
                "dense_missed_unstable_islands": comparison[
                    "narrow_unstable_islands_missed"
                ],
                "total_classifier_evaluations": run["summary"][
                    "total_classifier_evaluations"
                ],
            }
        )
    narrow_frame = pd.DataFrame(narrow_rows)
    narrow_frame.to_csv(OUTPUT_DIR / "narrow_island_sentinel_test.csv", index=False)

    multi_summary_rows = []
    for plant_name, _plant in plants:
        for strategy in STRATEGIES:
            run = representative_runs[(plant_name, strategy)]
            summary = dict(run["summary"])
            summary["plant"] = plant_name
            multi_summary_rows.append(summary)
    multi_summary = pd.DataFrame(multi_summary_rows)
    multi_summary.to_csv(OUTPUT_DIR / "multi_plant_summary.csv", index=False)

    # Figures contain no new classifier evaluations and exclude plotting time.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    timing_frame["mean_search_seconds"].plot.bar(
        yerr=timing_frame["std_search_seconds"],
        capsize=4,
        ax=axes[0],
    )
    axes[0].set_title("Five-run mean search time (G1)")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xlabel("Strategy")
    timing_frame["total_classifier_evaluations_per_run"].plot.bar(ax=axes[1])
    axes[1].set_title("Mean classifier evaluations per run")
    axes[1].set_ylabel("Calls")
    axes[1].set_xlabel("Strategy")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "repeated_runtime_and_calls.png", dpi=160)
    plt.close(fig)

    dense_summary = (
        dense_frame.groupby(["plant", "strategy"], as_index=False)["grid_agreement"]
        .mean()
    )
    dense_pivot = dense_summary.pivot(
        index="plant", columns="strategy", values="grid_agreement"
    )
    ax = dense_pivot.plot.bar(figsize=(10, 5))
    ax.set_title("Agreement with independent dense Kp references")
    ax.set_ylabel("Mean grid agreement")
    ax.set_xlabel("Plant")
    ax.set_ylim(0, 1.01)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "dense_validation_agreement.png", dpi=160)
    plt.close(ax.figure)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        sentinel_frame["sentinel_count"],
        sentinel_frame["total_classifier_evaluations"],
        marker="o",
    )
    axes[0].set_title("Boundary-only evaluations vs sentinel count")
    axes[0].set_xlabel("Interior sentinel fractions")
    axes[0].set_ylabel("Classifier calls")
    axes[1].plot(
        sentinel_frame["sentinel_count"],
        sentinel_frame["dense_mean_grid_agreement"],
        marker="o",
    )
    axes[1].set_title("Dense-reference agreement vs sentinel count")
    axes[1].set_xlabel("Interior sentinel fractions")
    axes[1].set_ylabel("Mean grid agreement")
    axes[1].set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sentinel_density_tradeoff.png", dpi=160)
    plt.close(fig)

    multi_pivot = multi_summary.pivot(
        index="plant", columns="strategy", values="total_classifier_evaluations"
    )
    ax = multi_pivot.plot.bar(figsize=(10, 5))
    ax.set_title("Resolution-matched evaluations across existing plants")
    ax.set_ylabel("Classifier calls")
    ax.set_xlabel("Plant")
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "multi_plant_classifier_calls.png", dpi=160)
    plt.close(ax.figure)

    print("Fair benchmark outputs written to:", OUTPUT_DIR)
    print("Kp lattice spacing:", config.kp_spacing)
    print("Binary tolerance:", config.binary_tolerance)
    print("\nRepeated G1 timing summary:")
    print(timing_frame.to_string())
    print("\nSentinel density summary:")
    print(sentinel_frame.to_string(index=False))
    print("\nSynthetic narrow-island sentinel test:")
    print(narrow_frame.to_string(index=False))
    print("\nCommon Ki rows:")
    print(pd.DataFrame(common_rows).to_string(index=False))
    print("\nDense-grid validation rows:", len(dense_frame))
    print("Multi-plant runs:", len(multi_summary))


if __name__ == "__main__":
    main()

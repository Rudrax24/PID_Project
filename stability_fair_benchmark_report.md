# PID Stability Search: Fair Benchmark and Validation

## Executive summary

This stage adds a separate, resolution-matched benchmark without changing the existing search algorithms, PID model, stability classifier, largest-rectangle routine, defaults, or original result files. New CSVs and plots are in [`benchmark_fair/`](./benchmark_fair/).

On the primary G1 experiment, the expanded-interval strategy was the least expensive under matched spatial resolution: 6,942 classifier evaluations and a mean search time of 23.484 s. Boundary-only used 7,448 evaluations and 25.126 s, so it was not faster than expanded-interval in this experiment. It did use fewer evaluations than exhaustive (8,011), but that advantage came with empirical reconstruction risk: a synthetic narrow unstable island remained undetected even after a fallback and periodic full-row scan.

The independent dense reference agreed with all three strategies on the 45 selected real-plant rows. This is finite-grid evidence only; it is not a guarantee against smaller or unsampled features. Consequently, boundary-only should remain experimental rather than be treated as a generally reliable replacement.

## What changed and what was preserved

- Added the fair benchmark sampler, runner, focused tests, CSVs, and plots under [`benchmark_fair/`](./benchmark_fair/).
- Added this report.
- The runner's timing rounds use a rotated strategy order to reduce fixed-order timing bias.
- The pre-existing `stability_search.py`, `is_stable()`, PID model, `find_largest_stable_rectangle()`, defaults, original CSVs, plots, and notebook backup were not modified by the fair benchmark.
- The fair runner imports definitions from notebook cells 0–3. **The complete notebook was not executed top-to-bottom**, so this report does not claim full notebook reproducibility.

## Why the original benchmark was not resolution-matched

The original implementation applies `coarse_steps=40` independently to each searched interval. Thus, 40 samples cover a narrow interval and also 40 samples cover a wide interval. The number of evaluations per interval can match even though the physical Kp spacing differs. In the preserved original results, exhaustive and expanded-interval each made 8,058 calls, but their sampled Kp coordinates were not thereby guaranteed to have the same spacing.

The separate fair benchmark instead uses:

- Global Kp limits: `[-10, 10]`.
- A shared 40-point full-range lattice, with spacing `20/39 = 0.5128205128`.
- The same lattice coordinates for every strategy. Each strategy evaluates only the lattice locations inside its permitted search regions; region endpoints are also included.
- A common absolute binary-refinement tolerance of `4.89063752e-7` Kp units, derived as the lattice spacing divided by `2^20`.
- The same Ki rows, plant, simulation settings, classifier, and initial conditions for all three policies.

This removes interval-width-dependent coarse spatial resolution. It does **not** make the original results interchangeable with the new results: the fair harness has a different sampling/refinement setup, and its counts should only be compared within the fair benchmark.

## Preserved original implementation benchmark

The legacy run is retained separately in [`benchmark_fair/original_implementation_benchmark.csv`](./benchmark_fair/original_implementation_benchmark.csv). This is the original benchmark, not a resolution-matched rerun.

| Strategy | Classifier calls | Coarse calls | Binary calls | Search time (s) | Rectangle time (s) | Agreement with exhaustive |
|---|---:|---:|---:|---:|---:|---:|
| Exhaustive | 8,058 | 6,000 | 2,058 | 27.002 | 0.0174 | 1.000000 |
| Expanded interval | 8,058 | 6,000 | 2,058 | 26.855 | 0.0193 | 0.999932 |
| Boundary-only | 7,434 | 4,762 | 2,436 | 25.337 | 0.0190 | 0.999864 |

The legacy exhaustive and expanded strategies had identical call counts because the implementation used the same fixed number of coarse samples for each searched interval, not a fixed absolute Kp spacing. Boundary-only made fewer calls in this run, but its exhaustive-map agreement was not perfect: the preserved CSV records one unstable-as-stable grid point and one stable-as-unstable grid point. These legacy results are kept distinct from the fair-lattice validation below.

## Resolution-matched results and repeated timing

The runner performed five G1 repetitions per strategy. Strategy order was rotated each round. Search time excludes plotting and is reported separately from the largest-rectangle calculation; the rectangle implementation itself was not changed. The timings are wall-clock measurements and can still be affected by external system load.

| Strategy | Calls/run | Coarse/run | Binary/run | Sentinel/run | Fallback full-row evals/run | Periodic full-row evals/run | Mean search (s) | SD (s) | Median (s) | Mean rectangle (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exhaustive | 8,011 | 6,000 | 2,011 | 0 | 0 | 0 | 27.491 | 1.296 | 26.945 | 0.0195 |
| Expanded interval | 6,942 | 4,931 | 2,011 | 0 | 0 | 0 | 23.484 | 0.139 | 23.442 | 0.0187 |
| Boundary-only (3 sentinels) | 7,448 | 4,845 | 2,367 | 236 | 40 | 728 | 25.126 | 0.355 | 24.957 | 0.0201 |

The total classifier evaluations across the five runs were 40,055 exhaustive, 34,710 expanded-interval, and 37,240 boundary-only. Fallback and periodic full-row evaluations are subsets of the coarse-evaluation count, not additional counts; they are broken out separately because those scans are especially relevant to boundary-only's cost.

For G1, expanded-interval used 1,069 fewer calls/run than exhaustive and 506 fewer than boundary-only. Boundary-only was about 8.1% slower than expanded-interval by mean search time in these measurements. Its mean remained faster than exhaustive, though the exhaustive timings had greater variability. Rectangle times are small and should not be combined with search timings when comparing search cost.

Detailed runs and summary statistics are in [`repeated_run_timings.csv`](./benchmark_fair/repeated_run_timings.csv) and [`repeated_timing_summary.csv`](./benchmark_fair/repeated_timing_summary.csv). The timing plot is [`repeated_runtime_and_calls.png`](./benchmark_fair/repeated_runtime_and_calls.png).

## Complete Ki-row accounting

All attempted rows, including empty and terminal rows, are recorded in [`ki_row_accounting.csv`](./benchmark_fair/ki_row_accounting.csv). The `common_ki_rows.csv` file records coordinate overlap across strategies.

| Plant | Strategy | Attempted rows | Retained rows | Empty rows | Terminating rows | Common attempted Ki coordinates |
|---|---|---:|---:|---:|---:|---:|
| G1 | Each of the three strategies | 150 | 49 | 101 | 1 | 150 |
| G5 | Each of the three strategies | 119 | 18 | 101 | 1 | 119 |
| G6 | Each of the three strategies | 116 | 15 | 101 | 1 | 116 |

The terminal empty row is included in the empty-row total. “Retained” means a row with at least one detected stable interval; empty rows were not silently dropped from accounting. All strategies attempted the same Ki coordinates within each plant experiment.

## Independent dense-grid validation

The reference is a separate uniform Kp scan using the existing simulation-based classifier. It uses 391 points over the global Kp range, with spacing `0.0512820513` (ten times finer than the fair coarse lattice). Five selected Ki rows were checked for each of G1, G5, and G6, giving 45 strategy/plant/row comparisons.

| Result across 45 comparisons | Finding |
|---|---:|
| Dense-grid agreement | 100% |
| Missed stable dense-grid points | 0 |
| False stable dense-grid points | 0 |
| Missed unstable islands in selected real-plant rows | 0 |
| Maximum left-boundary difference | `3.42345e-7` Kp |
| Maximum right-boundary difference | `2.68985e-7` Kp |
| Interval-count mismatches in selected real-plant rows | 0 |

The underlying per-row results, including interval counts and boundary errors, are in [`dense_grid_validation.csv`](./benchmark_fair/dense_grid_validation.csv); the plot is [`dense_validation_agreement.png`](./benchmark_fair/dense_validation_agreement.png).

This reference is finite-resolution, not mathematical ground truth. An island narrower than the dense spacing, or one located between tested Ki rows, can remain undetected. The comparison validates the sampled points and selected rows only; it does not prove equivalence of the maps.

## Boundary-only sentinel-density experiment

The real-plant comparison used G1 with configured sentinel counts 3, 5, and 9; other than sentinel count, settings were held constant. All three configurations achieved 100% agreement on the selected dense-reference rows. However, additional sentinels increased total evaluations and triggered more full-row fallbacks:

| Configured sentinels | Actual sentinel evaluations | Total calls | Fallback scans | Fallback evaluations | Periodic full-row validations | Dense agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 236 | 7,448 | 1 | 40 | 9 | 100% |
| 5 | 324 | 7,570 | 2 | 122 | 9 | 100% |
| 9 | 498 | 7,779 | 3 | 204 | 9 | 100% |

The actual evaluation count can be below the configured count because sentinel evaluation may stop after an observation triggers fallback. The full data is in [`sentinel_density_comparison.csv`](./benchmark_fair/sentinel_density_comparison.csv), with [`sentinel_density_tradeoff.png`](./benchmark_fair/sentinel_density_tradeoff.png).

### Narrow unstable island test

The focused synthetic case starts with a stable interval `[-2, 2]`; at the next Ki row the true stable regions are `[-2, -0.05]` and `[0.05, 2]`. With the sparse sentinel positions `0.25` and `0.75`, neither probe enters the central unstable gap, so the boundary-only reconstruction keeps one interval. The periodic full-row check also fails to find the gap because its 40-point lattice over `[-5, 5]` has spacing about `0.2564`, much wider than the `0.1` gap. The dense reference spacing is about `0.02564` and resolves two intervals.

| Sentinel configuration | Sentinel calls | Fallback scans | Periodic full-row checks | Adaptive intervals | Dense-reference intervals | False-stable dense points |
|---|---:|---:|---:|---:|---:|---:|
| Sparse positions 0.25, 0.75 | 4 | 0 | 1 | 1 | 2 | 3 |
| 3 evenly spaced | 3 | 1 | 1 | 1 | 2 | 3 |
| 5 evenly spaced | 4 | 1 | 1 | 1 | 2 | 3 |
| 9 evenly spaced | 6 | 1 | 1 | 1 | 2 | 3 |

The evenly spaced settings detect an unstable sentinel and trigger a fallback, but the fallback and periodic scan use the same insufficiently fine coarse lattice and still miss the island. Increasing sentinel density therefore raises cost in this case without correcting the reconstruction. See [`narrow_island_sentinel_test.csv`](./benchmark_fair/narrow_island_sentinel_test.csv).

## Multi-plant comparison

The same fair-lattice settings were run on three transfer functions already defined in the notebook: G1, G5, and G6. One measured run per plant/strategy is reported here; the five-repeat timing study above is for G1. All strategies attempted identical Ki rows within each plant.

| Plant | Strategy | Calls | Search time (s) | Fallback scans | Periodic full-row validations | Attempted rows |
|---|---|---:|---:|---:|---:|---:|
| G1: `6/((s+1)(s+2)(s+3))` | Exhaustive | 8,011 | 29.797 | 0 | 0 | 150 |
|  | Expanded interval | 6,942 | 23.609 | 0 | 0 | 150 |
|  | Boundary-only | 7,448 | 24.891 | 1 | 9 | 150 |
| G5: `120/((s+1)...(s+5))` | Exhaustive | 5,500 | 18.826 | 0 | 0 | 119 |
|  | Expanded interval | 4,992 | 17.124 | 0 | 0 | 119 |
|  | Boundary-only | 5,309 | 18.037 | 1 | 3 | 119 |
| G6: `720/((s+1)...(s+6))` | Exhaustive | 5,256 | 18.018 | 0 | 0 | 116 |
|  | Expanded interval | 4,826 | 16.633 | 0 | 0 | 116 |
|  | Boundary-only | 5,067 | 17.390 | 1 | 2 | 116 |

Boundary-only used fewer calls than exhaustive, but more than expanded-interval, on each of these plants. The selected-row dense-grid results were consistent across all three. Raw rows and summaries are available in [`multi_plant_runs.csv`](./benchmark_fair/multi_plant_runs.csv) and [`multi_plant_summary.csv`](./benchmark_fair/multi_plant_summary.csv); call counts are plotted in [`multi_plant_classifier_calls.png`](./benchmark_fair/multi_plant_classifier_calls.png).

## Known limitations and recommendations

1. **Boundary-only cannot guarantee detection of an interior feature.** Sentinel probes and finite full-row scans can both miss a narrow island if no sample lands inside it. The synthetic test demonstrates this failure even when fallback and periodic validation run.
2. **More sentinels do not guarantee correctness.** They increased evaluations and fallbacks in the real-plant benchmark; in the narrow-island case they caused fallback but the coarse fallback grid still missed the island.
3. **Dense validation has limited scope.** It checks selected Ki rows and a finite Kp grid. It cannot certify unsampled coordinates or prove stability-region topology.
4. **Runtime is variable.** Timing rounds are interleaved by rotating strategy order, but wall-clock results remain sensitive to machine load. The exhaustive standard deviation was notably higher than the other strategies in the five-run sample.
5. **The notebook was only partially executed.** The harness loaded required definition cells 0–3; it did not run every notebook cell or regenerate all original notebook outputs.
6. **Warnings from the pre-existing simulation stack appeared** for ill-conditioned or diverging trials during unstable regions. The classifier and model were not changed for this benchmark.

Keep the existing expanded-interval strategy as the practical comparison/default unless broader validation supports a change. Treat boundary-only as an experimental acceleration with explicit fallback/validation costs and known blind spots. For stronger assurance, make the fallback sampling resolution fine enough for the minimum feature size of interest, or periodically compare against a dense scan; neither step turns finite sampling into a proof.

## Reproducibility artifacts

- Runner: [`benchmark_fair/run_fair_benchmarks.py`](./benchmark_fair/run_fair_benchmarks.py)
- Fair sampler and validation utilities: [`benchmark_fair/fair_benchmark.py`](./benchmark_fair/fair_benchmark.py)
- Focused tests: [`benchmark_fair/test_fair_benchmark.py`](./benchmark_fair/test_fair_benchmark.py)
- Generated tables and plots: [`benchmark_fair/`](./benchmark_fair/)

The focused test file passed: **3 passed**. Python syntax compilation also passed for the fair benchmark modules. These checks do not substitute for a full notebook run.

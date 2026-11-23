# PID Stability Search: Strategy Comparison and Complexity

## Summary

Three selectable strategies are available in `stability_without_poles.ipynb`:

- `exhaustive`: scan the full Kp interval on each visited Ki row.
- `expanded_interval`: preserve the existing adaptive baseline; scan each previous stable interval expanded by `search_margin` (0.5 in the benchmark).
- `boundary_only`: scan neighborhoods of the previous interval boundaries, reuse the old interior subject to sentinels, and fall back to a full-row scan when its checks detect ambiguity.

The original notebook was copied to `stability_without_poles_original_backup.ipynb` before editing. The search implementation is in `stability_search.py`; the original simulation-based `is_stable()` and `find_largest_stable_rectangle()` implementations were not changed.

Boundary-only is an empirical optimization, not a stability guarantee. A finite set of sentinels cannot rule out a narrow unstable island between probes. Periodic full-row validations and fallback checks reduce that risk but do not eliminate it.

## Benchmark setup

The three strategies were run sequentially with the same experiment arguments:

| Parameter | Value |
|---|---:|
| Plant object | `G1 = TransferFunction([6], [1, 6, 11, 6])` |
| Kd | 0 |
| Kp range | [-10, 10] |
| Ki range | [-10, 10] |
| Seed Ki | 0 |
| Initial and refined Ki step | 0.1 |
| Expanded/boundary margin | 0.5 |
| Coarse full-row samples | 40 |
| Binary boundary iterations | 20 |
| Rectangle/map Kp resolution | 300 |
| Stability simulation | Existing classifier, 300 time units, dt=1 |
| Boundary-only sentinels | 3 interior fractions: 0.25, 0.5, 0.75, plus boundary anchors |
| Periodic full-row validation | Every 5 post-onset Ki rows |

**Model-name caveat:** the notebook labels this first plant `-6/(s+1)(s-2)(s+3)`, but the actual object passed to the search is `6/(s+1)(s+2)(s+3)`. The benchmark used the existing `G1` object unchanged, not the inconsistent display label.

The table below reports one measured run, not an average. Search time ends after the Ki-row searches; rectangle time is measured separately and added to total time. Plot rendering is excluded. The stability map comparison uses the 49 Ki rows retained in the returned result dictionaries and 300 Kp plot samples per row (14,700 grid points). The search processed 150 Ki rows, including rows with no stable intervals.

## Measured results

| Strategy | `is_stable()` calls | Coarse Kp | Binary | Sentinels | Ki rows processed | Full-row fallbacks | Periodic validations | Stable intervals | Search (s) | Rectangle (s) | Total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exhaustive | 8,058 | 6,000 | 2,058 | 0 | 150 | 0 | 0 | 49 | 27.002 | 0.017 | 27.019 |
| Expanded interval | 8,058 | 6,000 | 2,058 | 0 | 150 | 0 | 0 | 49 | 26.855 | 0.019 | 26.875 |
| Boundary-only | 7,434 | 4,762 | 2,436 | 236 | 150 | 2 | 9 | 49 | 25.337 | 0.019 | 25.356 |

On this run, boundary-only used **624 fewer classifier calls (7.7%)** and took **1.663 seconds less (6.2%)** than exhaustive. It took 1.518 seconds less than expanded-interval. The expanded-interval baseline used the same number of calls as exhaustive. In this implementation, `coarse_steps` is applied to each searched range; with one stable interval per row, narrowing that range does not by itself reduce the number of coarse calls. Boundary-only instead scales window samples to the full-row Kp spacing, then adds its sentinel, transition-refinement, fallback, and validation costs.

The observed improvement is specific to this plant and these settings. It is not evidence that boundary-only will be faster for multiple intervals, more frequent fallbacks, or other plants. Only one timed run was taken, so small wall-clock differences should not be treated as a stable performance estimate.

## Agreement with the baselines

All three runs returned 49 stable intervals, and none had a differing interval count on any returned Ki row. Measured map/boundary/rectangle comparisons:

| Boundary-only compared with | Map agreement | Unstable-as-stable grid points | Stable-as-unstable grid points | Mean paired edge difference (Kp) | Max paired edge difference (Kp) | Largest rectangle |
|---|---:|---:|---:|---:|---:|---|
| Exhaustive | 99.9864% | 1 | 1 | 0.001214 | 0.075343 | Exact match |
| Expanded interval | 99.9932% | 0 | 1 | 0.000445 | 0.043578 | Exact match |

The largest rectangle had zero area and boundary differences against both baselines at the configured 300-column resolution. The map agreement is based on the interval-derived grid, not a new simulation at all 14,700 plotted points. `unstable-as-stable` is the more safety-relevant disagreement: one plotted grid point was classified stable by boundary-only but unstable by exhaustive.

The generated comparison table and plots are:

- `stability_strategy_comparison.csv`
- `stability_boundary_validation.csv`
- `stability_strategy_comparison.png`
- `stability_map_strategy_comparison.png`

## Validation coverage and limitations

The focused tests use deterministic stability predicates to exercise search logic; those are algorithm tests, not additional PID simulation experiments.

| Situation | Validation performed | Result |
|---|---|---|
| Approximately continuous, bell-shaped region | Actual `G1` simulation benchmark and three strategy maps | Compared against both baselines; small grid/boundary differences reported above |
| Expanding and contracting intervals | Synthetic predicate tests | Reconstructed both without fallback |
| Multiple disjoint intervals | Synthetic predicate test | Reconstructed both intervals without fallback |
| Interval disappearance | Synthetic predicate test | Sentinel failure triggered a full-row scan |
| Split observed by sentinels | Synthetic predicate test | Full-row fallback recovered two intervals |
| Split missed by sparse sentinels | Synthetic predicate test with a narrow interior gap | Demonstrated that boundary-only can reuse an interval that is not actually continuous |
| Split detected by periodic validation | Synthetic predicate test | Full-row validation replaced the reconstructed interval with two observed intervals |
| Positive and negative Ki directions; global Kp limits | Focused tests | Both directions and clipping behavior exercised |

Sentinel count and fractional placements are configurable using `sentinel_count` and `sentinel_positions`. Boundary anchors are additionally evaluated. A sentinel that becomes unstable, a stable run extending beyond a non-global window edge, a newly observed disconnected run, or an inconsistent reconstruction triggers a full-row fallback. A periodic full-row scan replaces the boundary result if the intervals disagree beyond the configured coarse-window tolerance.

These checks do **not** make the reconstruction complete: narrow unstable gaps between sentinels and coarse boundary samples can remain unseen. Periodic scans only detect changes visible to their own finite coarse grid and occur at a configurable frequency. The returned map therefore remains an adaptive estimate, and the benchmark confirms one small disagreement with exhaustive search.

## Three generations of the search

Let:

- `R` be the number of visited Ki rows;
- `P` be the number of full-row coarse Kp samples;
- `I_r` be the number of intervals on row `r`;
- `B` be the number of binary iterations per transition;
- `C_sim` be the cost of one call to the existing simulation classifier.

These are theoretical estimates; they are distinct from the measured counts above.

### Version 1 - exhaustive search

Every visited Ki row is scanned across all of [Kp_min, Kp_max], then observed transitions are refined by binary search.

- **Workflow:** full-row coarse sampling, interval grouping, binary refinement.
- **Why:** baseline with no prior-row locality assumption.
- **Cost:** approximately `O(R * (P + 2 I_r B) * C_sim)`.
- **Advantages:** searches the full Kp domain on each visited row; does not miss an interval merely because it lies outside a previous row's interval.
- **Limitations:** most classifier calls; still limited by the finite coarse grid and the existing finite-horizon classifier.
- **Measured:** 8,058 classifier calls, 27.019 seconds total.

### Version 2 - reduced/expanded interval search

The initial row and rows before stability are scanned fully. After stability is found, each previous interval is expanded on both sides by 0.5 and the whole expanded interval is scanned.

- **Workflow:** reuse previous-row intervals only to select the next row's search ranges; classify every coarse point in each expanded range.
- **Why:** exploit empirically local boundary motion while retaining a full scan of each selected range.
- **Cost:** approximately `O(sum_r (S_r + 2 I_r B) * C_sim)`, where `S_r` is `coarse_steps` per searched range. Because this implementation uses a fixed number of samples per range, reducing a single range's width does not necessarily reduce call count.
- **Advantages:** simple baseline; detects changes anywhere within each expanded interval.
- **Limitations:** a newly stable interval outside the expanded ranges can be missed; multiple intervals can require more work; does not scan Kp gaps between selected ranges.
- **Measured:** 8,058 calls, 26.875 seconds total, 99.9932% plotted-map agreement with exhaustive on retained rows.

### Version 3 - boundary-only search

The initial row remains exhaustive. For each later row, the algorithm evaluates multiple interior sentinels and scans merged windows around both boundaries of each previous interval. It reuses the remainder of the old interval, refines observed transitions, and reconstructs the new interval. Ambiguous observations trigger a full-row scan; periodic full-row checks guard against some undetected changes.

- **Workflow:** sentinel checks -> merged boundary windows -> transition refinement -> reconstruction checks -> optional fallback/full-row validation.
- **Why:** reduce repeated simulations of an interior expected to remain stable.
- **Cost:** approximately `O((sum_r (Q_r + W_r + 2 I_r B) + (F + V - M) P) * C_sim)`, where `Q_r` is the sentinel count, `W_r` the coarse samples in merged boundary windows, `F` the number of full-row fallbacks, `V` the number of periodic validations, and `M` the validations that also counted as a fallback because they found a mismatch.
- **Advantages:** can reduce coarse calls when intervals are wide relative to the boundary windows; supports multiple intervals and growth/shrinkage; fallback checks are explicit and counted.
- **Limitations:** interior stability between sentinels is assumed, not proven; unseen new regions away from all searched windows can be missed; periodic full-row checks add cost and remain coarse; fallback can erase the savings.
- **Measured:** 7,434 calls, 25.356 seconds total, two fallback scans and nine periodic validations. It had one stable-as-unstable and one unstable-as-stable grid disagreement against exhaustive.

All strategies use the same transfer function, classifier, Ki row schedule, simulation settings, coarse full-row count, and binary iteration count in the benchmark. The boundary method's local window sample count is proportional to window width so its coarse Kp spacing follows the same full-row grid spacing; it is not separately tuned to the measured outcome.

## Largest stable rectangle

The original top/bottom row-intersection implementation is retained in the notebook as `find_largest_stable_rectangle_reference()`. The notebook's `find_largest_stable_rectangle()` now uses a coordinate-aware histogram and monotonic stack, implemented in `largest_stable_rectangle.py`.

Both functions build the same Boolean `H x W` stability matrix from the supplied Ki rows and Kp samples. For each bottom Ki row, the optimized function tracks each Kp column's earliest connected stable row, with the physical histogram height `Ki[bottom] - Ki[start]`. It resets this state on a false cell or when an adjacent supplied Ki gap is strictly greater than `max_ki_gap`. A monotonic stack identifies Kp spans sharing the minimum height; widths and heights are computed from the actual endpoint coordinates. Equal-area candidates use `(top row, bottom row, left column, right column)` ordering to preserve the reference's first-wins behavior. The result dictionary remains `Kp_range`, `Ki_range`, `area`, and `center`.

- **Reference search time:** `O(H^2 W)` worst case, after matrix construction. Each top/bottom pair performs width-`W` intersection and contiguous-run scanning.
- **Optimized search time:** `O(HW)` after matrix construction. Each row visits every Kp column, and each histogram bar is pushed and popped at most once in its row's stack.
- **Matrix construction:** for `M` total interval entries, painting the matrix takes `O(WM)`; initialization takes `O(HW)`. Thus the whole optimized function is `O(HW + WM)`, or `O(HW)` when the number of intervals per Ki row is bounded.
- **Memory:** both retain the `O(HW)` Boolean matrix. The optimized search adds `O(W)` histogram and stack storage.
- **Missing rows:** neither function infers Ki rows absent from `results`. An explicitly supplied empty row is all false; otherwise the existing adjacent-gap rule is applied only to supplied keys.
- **Area semantics:** as before, rectangle bounds are sample-coordinate endpoints, not cell edges, and zero-area candidates do not replace the result.

The focused reference-equivalence tests cover structured, gap, boundary, tie, degenerate, and 400 deterministic randomized cases. A separate 10-repeat search-kernel benchmark (matrix construction excluded) is recorded in `benchmark_fair/largest_rectangle_performance.csv`. It found speedups from `0.44x` to `7.60x`: the optimized Python stack was slower on the small cached G5/G6 maps, but faster on G1 and the larger synthetic matrices. The benchmark validates the public APIs against the same prebuilt-matrix kernels before timing.

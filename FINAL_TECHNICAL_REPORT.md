# PID Controller Stability Mapping: Final Technical Report

## Executive summary

This project maps simulated closed-loop behavior over proportional and integral
controller gains, \(K_p\) and \(K_i\), for six transfer-function plants. It
compares three ways to search the gain plane and two implementations for
extracting the largest axis-aligned stable rectangle.

The main findings are:

- The stability decision is a finite-horizon, numerical step-response test,
  not a mathematical proof of stability.
- In the resolution-matched G1 benchmark, expanded-interval search was the
  best practical strategy measured: it averaged 6,942 classifier evaluations
  and 23.484 seconds, compared with 8,011 evaluations and 27.491 seconds for
  exhaustive search.
- Boundary-only search was faster than exhaustive on that benchmark, but
  slower than expanded-interval and is vulnerable to interior features between
  its probes. It remains experimental.
- The coordinate-aware histogram/monotonic-stack rectangle search has
  \(O(HW)\) search complexity, versus \(O(H^2W)\) for the reference search.
  Public-function timings show that it is faster on G1 and the larger synthetic
  cases but slower on the smaller G5 and G6 cases.
- Rectangle extraction is a very small part of the recorded stability-search
  runtime. Reducing classifier and simulation work is more consequential for
  total runtime.

The complete notebook has **not** been executed from beginning to end. All
timings and validation results below refer to the saved benchmark artifacts
listed in [Sources and evidence](#sources-and-evidence); this documentation
work did not rerun searches, regenerate results, or modify implementation
files.

## 1. Project overview and simulation-based stability decision

The goal is to identify regions in the \(K_p\)-\(K_i\) plane where a PID
controller, paired with a plant, passes the project's response classifier.
Mapping both gains is useful because a setting can pass at one combination and
fail nearby; a single tuned pair does not show the extent or shape of the
acceptable region.

The six plants are represented in
[stability_without_poles.ipynb](./stability_without_poles.ipynb) using
`python-control` transfer functions whose numerator and denominator arrays
contain descending polynomial coefficients:

| Plant | Transfer function |
|---|---|
| G1 | \(6/((s+1)(s+2)(s+3))\) |
| G2 | \(2(s+2)/(s^3-3s^2+2s+4)\) |
| G3 | \(1/(s^4+5s^3+2s^2-7s+1)\) |
| G4 | \(12(s+2)/((s-1)(s-2)(s+3)(s+4))\) |
| G5 | \(120/\prod_{j=1}^{5}(s+j)\) |
| G6 | \(720/\prod_{j=1}^{6}(s+j)\) |

For each trial, `is_stable(Kp, Ki, Kd, plant)` builds the controller transfer
function \((K_d s^2+K_p s+K_i)/s\), forms its unity-feedback closed loop with
the plant, and simulates a unit-step response. The default horizon is 300
time-units sampled at `dt=1`, giving samples from 0 through 300. With the
benchmark's `Kd=0`, the proportional and integral terms are still represented
by the same controller construction.

The classifier returns `False` if the response contains non-finite values, has
the wrong length, exceeds the absolute output bound of 10, or fails any of
these final-response checks:

1. Estimate the final value as the mean of the last 10% of samples.
2. Set tolerance to
   \(\max(0.01\max(|y_{\mathrm{final}}|,1),10^{-5})\).
3. Require the final-tail range to be no more than twice that tolerance.
4. Require the difference between the first and last quarter means of the tail
   to be no more than the tolerance.
5. Require the last sample outside the tolerance band to occur before 90% of
   the simulation horizon.

Specified numerical and control-library exceptions also produce `False`.
Therefore, “stable” in this report means “passed this finite simulation and
threshold test.” It does not establish asymptotic stability, nor does the
finite horizon rule out slow divergence beyond the simulated time.

An **`is_stable()` evaluation** is one complete simulation-based classifier
call. A **coarse evaluation** is such a call at a point in the initial Kp
sampling grid. A **binary-refinement evaluation** is a call at a midpoint
between a sampled stable point and unstable point to localize an observed
transition. A **Ki row** is one fixed \(K_i\) coordinate with a corresponding
Kp scan. A **stable interval** is a contiguous Kp interval reconstructed from
stable samples and refined transitions. **Boundary refinement** is the
midpoint process used to estimate one edge of that interval.

## 2. Search workflow and computational cost

The notebook seeds the map at \(K_i=0\) and fully scans the initial Kp range.
It then searches positive and negative Ki directions independently from that
seed. Before a direction finds a stable row, it advances using the initial Ki
step; after stability is found, it uses the refined Ki step. If a direction
has not yet found stability, an empty row does not stop that direction. After
stability has been found, the first empty row terminates that direction.
Results are retained as a mapping from supplied Ki coordinates to lists of
stable Kp intervals. Missing coordinates are not inferred.

The six-plant notebook cell and the fair benchmark use these matched search
settings: \(K_p,K_i\in[-10,10]\), seed \(K_i=0\), initial and refined Ki steps
of 0.1, search margin 0.5, 40 coarse points over the global Kp range, 20
binary iterations, and 300 Kp points for rectangle/map construction. The
classifier uses the defaults described above. These experiment arguments
differ from some callable defaults in the notebook function; benchmark
comparisons use the explicit experiment configuration, not the function
defaults.

For a full-row scan, coarse samples are classified, contiguous stable runs are
identified, and each observed stable/unstable edge is refined. If \(R\) is the
number of attempted Ki rows, \(P\) the number of Kp coarse samples per full
row, \(I_r\) the number of stable intervals on row \(r\), and \(B\) the number
of binary iterations per transition, the classifier-call work is on the order
of

\[
  \sum_{r=1}^{R}\left(P + 2I_r(B+1)\right)
\]

in a full-row scan, with fewer than two transitions for intervals that touch
the global Kp limits. Here, \(P\) is the number of coarse classifier
evaluations on that row. In `stability_search.py`, `refine_boundary()` first
classifies `low` once, then classifies one midpoint on each of its `B`
iterations. That is \(B+1\) evaluations per refined edge; at most two edges
per interval give the \(2I_r(B+1)\) term. This is an approximate upper-bound
accounting, since intervals touching scan-range endpoints need fewer
refinements. The fair harness uses its own tolerance-driven refinement loop;
its classifier-call figures below are measured counts, not estimates from
this fixed-iteration formula. Each call also incurs the cost of constructing
and simulating a closed-loop response, so evaluation counts are not the only
runtime factor. Exhaustive search repeats full-row coarse sampling for every
Ki row and is consequently expensive.

## 3. The three search strategies

The policies are implemented in [stability_search.py](./stability_search.py)
and selected by the notebook's `search_strategy` argument.

### A. Exhaustive

The initial row and every subsequent Ki row are scanned across the full
global Kp range. This is the no-locality-assumption comparison baseline: it
does not omit a Kp region merely because it was not near a previous interval.
It remains limited by the finite Kp sampling grid and by the simulation
classifier. Its coarse work is approximately \(R P\), plus boundary
refinement.

### B. Expanded interval

After the first detected stable row, each prior stable interval \([L,R]\) is
expanded to approximately \([L-0.5,R+0.5]\), clipped to global Kp limits.
The entire selected range is sampled on the next Ki row, not just its edges.
The ranges are merged if they overlap, then observed interval edges are
refined. Before stability is found, the search uses full rows.

This was the notebook's adaptive comparison baseline. It avoids many Kp
locations far from previously detected intervals, but can miss a newly stable
region outside all expanded ranges. In the original implementation,
`coarse_steps` was applied independently to each searched interval, so a
narrow range could receive the same number of samples as the full domain.
Equal call counts therefore did not imply equal Kp spacing.

### C. Boundary-only

The initial Ki row is still fully scanned. For later rows, the algorithm
places windows around both boundaries of each prior stable interval, merges
overlapping windows, samples them, and refines observed transitions. It also
checks interior sentinels and boundary anchors, then reuses the previous
stable interior when observations support that reconstruction. Ambiguous
results trigger full-row fallback scans. Periodic full-row validation is
configured every five post-onset rows in the fair benchmark.

This reduces repeated classification of an expected-stable interior, but that
expectation is empirical. A finite set of sentinels cannot prove the whole
interior remains stable; an unstable island between probe points can be
missed. Fallback and periodic scans only find features that their own finite
sampling resolves. The deliberately constructed narrow-island test below
demonstrates this limitation; it was a synthetic failure case, not an
observation from the real plant experiments.

## 4. Fair search benchmarks

### Resolution matching

The preserved legacy comparison is in
[original_implementation_benchmark.csv](./benchmark_fair/original_implementation_benchmark.csv).
Its `coarse_steps=40` were distributed separately over each searched
interval. Thus, equal sample counts could represent different absolute Kp
spacings. That legacy benchmark is useful as a record of the original
implementation, but it is not a resolution-matched comparison.

The separate fair harness in
[fair_benchmark.py](./benchmark_fair/fair_benchmark.py) uses a shared global
Kp lattice for every strategy:

- Global Kp limits: \([-10,10]\).
- Forty global lattice points, spaced by \(20/39 \approx 0.512820513\).
- A strategy evaluates only the lattice locations inside its permitted
  ranges; each permitted range's endpoints are also included.
- Binary refinement stops at one absolute tolerance,
  \(0.512820513/2^{20}\approx4.89063752\times10^{-7}\) Kp units.
- The same plant, classifier, Ki coordinates, simulation settings, and
  experiment parameters are used for each strategy.

This fixes spatial-resolution comparability for the fair experiment without
changing the original search algorithms or rewriting the legacy results.

### Preserved legacy benchmark

The historical results below are from the original implementation
comparison, not the shared-lattice rerun:

| Strategy | Classifier calls | Search time (s) | Map agreement with exhaustive |
|---|---:|---:|---:|
| Exhaustive | 8,058 | 27.002 | 100.000% |
| Expanded interval | 8,058 | 26.855 | 99.993% |
| Boundary-only | 7,434 | 25.337 | 99.986% |

The legacy table's boundary-only comparison had one unstable-as-stable and
one stable-as-unstable plotted-grid point relative to exhaustive. The
expanded-interval legacy run had one unstable-as-stable point. These legacy
results must not be mixed with the resolution-matched results below.

### Resolution-matched G1 results

Five timed G1 searches were performed per strategy. Strategy order was
rotated between runs. Search timing excludes plotting and reports rectangle
extraction separately. Standard deviations and other statistics are from
[repeated_timing_summary.csv](./benchmark_fair/repeated_timing_summary.csv).
The coarse count includes classifier evaluations made during fallback and
periodic full-row scans. The fallback and periodic columns are diagnostic
subsets of that coarse count, not additional classifier calls to add again to
the total; a periodic scan that detects a mismatch can be represented in both
diagnostic counts. Sentinel evaluations are reported separately and are not
included in coarse evaluations.

| Strategy | Calls/run | Coarse | Binary | Sentinels | Fallback full-row evals | Periodic full-row evals | Mean search (s) | SD (s) | Median (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exhaustive | 8,011 | 6,000 | 2,011 | 0 | 0 | 0 | 27.491 | 1.296 | 26.945 |
| Expanded interval | 6,942 | 4,931 | 2,011 | 0 | 0 | 0 | 23.484 | 0.139 | 23.442 |
| Boundary-only (3 sentinels) | 7,448 | 4,845 | 2,367 | 236 | 40 | 728 | 25.126 | 0.355 | 24.957 |

Each G1 strategy processed 150 attempted Ki rows (49 retained stable rows,
101 empty rows including the terminating row). Each strategy's attempted Ki
coordinates matched the others.

Expanded interval used 1,069 fewer classifier calls per run than exhaustive
(about 13.3%) and averaged about 14.6% less search time. Boundary-only used
563 fewer calls than exhaustive (about 7.0%) and averaged about 8.6% less
search time. Compared with expanded interval, boundary-only used 506 more
calls (about 7.3%) and averaged about 7.0% more search time. Although
boundary-only made 86 fewer coarse evaluations than expanded interval, its
additional binary and sentinel evaluations more than offset that difference.
Fallback and periodic validation work adds further coarse evaluations.

The expanded-interval strategy was the best measured practical strategy for
this fair G1 benchmark. This is a result for the tested plant and settings,
not a guarantee for other maps.

### Dense-grid validation

The independent reference scanned 391 uniform Kp points over the global range,
with spacing approximately 0.0512820513. Five selected Ki rows were checked
for each of G1, G5, and G6, comparing all three strategies: 45
plant/strategy/row comparisons in total. The saved
[dense_grid_validation.csv](./benchmark_fair/dense_grid_validation.csv)
records:

| Measure across selected rows | Result |
|---|---:|
| Grid agreement | 100% |
| Missed stable dense-grid points | 0 |
| False stable dense-grid points | 0 |
| Interval-count mismatches | 0 |
| Maximum left-boundary difference | \(3.42345\times10^{-7}\) Kp |
| Maximum right-boundary difference | \(2.68985\times10^{-7}\) Kp |

This is finite-resolution evidence for selected rows only. It cannot rule out
features narrower than the dense spacing or features between selected Ki
rows, and it is not a mathematical proof of map equivalence.

### Sentinel density and the narrow-island failure

On the G1 sentinel-density experiment, only sentinel count was varied; other
settings remained fixed. The dense-reference agreement was 100% on the
selected real-plant rows for all three settings.

| Sentinels configured | Sentinel evaluations | Total calls | Fallback scans | Fallback evaluations | Periodic validations | Search time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 236 | 7,448 | 1 | 40 | 9 | 24.891 |
| 5 | 324 | 7,570 | 2 | 122 | 9 | 25.398 |
| 9 | 498 | 7,779 | 3 | 204 | 9 | 25.969 |

The 3-sentinel result's 24.891 seconds is the single-run datum shown in the
separate sentinel-density comparison; the harness reuses the representative
3-sentinel G1 run for that row. It is not the five-run repeated-G1 mean or
median; those are 25.126 seconds and 24.957 seconds, respectively, in the
resolution-matched G1 table.

In the constructed failure case, the next row's true regions were
\([-2,-0.05]\) and \([0.05,2]\), leaving a 0.1-wide unstable gap. Sparse
sentinels at fractions 0.25 and 0.75 did not land in that gap. The periodic
40-point full-row scan over \([-5,5]\) had spacing about 0.2564, also too
coarse to resolve it. A 391-point dense reference had spacing about 0.02564
and found two intervals.

Increasing sentinel density caused a fallback in the evenly spaced cases,
but the fallback's coarse lattice still missed the island. The saved
[narrow_island_sentinel_test.csv](./benchmark_fair/narrow_island_sentinel_test.csv)
shows one reconstructed interval versus two dense-reference intervals and
three false-stable dense points for each tested configuration. Total calls
rose from 296 with the specified sparse probes to 323, 324, and 326 with 3,
5, and 9 evenly spaced probes. More probes therefore increased cost without
fixing this particular failure.

### Multi-plant comparison

The same fair settings were used for G1, G5, and G6. G1 is represented by the
first run in the repeated G1 set; G5 and G6 have one measured run per
strategy, not five-run timing samples.

| Plant | Strategy | Calls | Search time (s) |
|---|---|---:|---:|
| G5 | Exhaustive | 5,500 | 18.826 |
| G5 | Expanded interval | 4,992 | 17.124 |
| G5 | Boundary-only | 5,309 | 18.037 |
| G6 | Exhaustive | 5,256 | 18.018 |
| G6 | Expanded interval | 4,826 | 16.633 |
| G6 | Boundary-only | 5,067 | 17.390 |

Expanded interval used the fewest calls and took the least time in both
single-run comparisons. Boundary-only used fewer calls than exhaustive, but
more than expanded interval, for both plants. Because G5 and G6 have one
timing run per strategy, these timings are descriptive rather than reliable
estimates of repeated-run variability.

## 5. Largest stable rectangle

The rectangle is the largest axis-aligned rectangle whose sampled Kp/Ki
matrix cells are all classified stable. It provides a compact region and
center point for reporting candidate gains; it does not assert that every
continuous gain value between samples has been simulated.

### Reference implementation

`find_largest_stable_rectangle_reference()` is retained in
[stability_without_poles.ipynb](./stability_without_poles.ipynb). It:

1. Sorts the supplied Ki keys and creates `plot_resolution` equally spaced Kp
   coordinates.
2. Builds an \(H\times W\) Boolean matrix by painting each row's supplied
   stable intervals onto the sampled Kp coordinates.
3. Enumerates each possible top Ki row and incrementally ANDs its stable
   columns with each successive bottom row.
4. Stops extending a candidate when an adjacent supplied Ki gap is strictly
   greater than `max_ki_gap`.
5. Finds contiguous true runs in the row intersection and calculates
   physical area from endpoint differences.
6. Keeps the first candidate encountered when areas tie, since it replaces
   the current result only when area is strictly greater.

### Coordinate-aware histogram and monotonic stack

The optimized function in
[largest_stable_rectangle.py](./largest_stable_rectangle.py) uses the same
matrix construction, then updates a histogram for every bottom Ki row. At Kp
column \(j\), the histogram tracks the physical Ki span from the earliest
connected stable row to the current bottom row:

\[
h_j = K_i[\text{bottom}] - K_i[\text{start}_j].
\]

This is a coordinate-aware height, not a row count. A false matrix cell resets
that column's height and start row. If an adjacent supplied Ki gap is
**strictly greater** than `max_ki_gap`, all histogram state resets; a gap
equal to the threshold remains allowed, matching the reference.

For a fixed bottom row and horizontal interval \([l,r]\), the limiting
histogram height is

\[
h_{\min} = \min_{j\in[l,r]} h_j.
\]

Since Ki coordinates are ordered, this minimum corresponds to the latest
per-column stable start and therefore to the earliest row shared by every
column in the interval. It is the greatest vertical span for which all cells
in that horizontal interval remain stable and connected without crossing a
prohibited Ki gap.

A monotonic stack finds the maximal horizontal span over which each bar can be
the minimum. When a lower bar or the final sentinel is reached, the popped
bar yields a candidate rectangle. Kp width uses the actual endpoint
coordinates, `Kp[right] - Kp[left]`; Ki height uses `Ki[bottom] - Ki[top]`.
Thus area is the physical coordinate product, not the count of matrix cells.
False cells split disconnected intervals, and the sentinel flushes candidates
that reach the last Kp column.

The optimized function retains the result fields `Kp_range`, `Ki_range`,
`area`, and `center`. It explicitly resolves equal-area candidates by
`(top_row_index, bottom_row_index, left_column_index, right_column_index)`,
matching the reference's first-wins traversal order. Both implementations
return `None` when no positive-area rectangle exists. Neither infers missing
Ki rows; only supplied keys and the existing adjacent-gap rule are used.

### Equivalence reasoning

For a chosen bottom row, each positive histogram bar gives a Kp column's
longest connected stable vertical span ending at that row. A rectangle is
valid exactly when every column in its horizontal interval has a span at
least as tall as the rectangle. The minimum bar height across that interval
therefore sets the maximum valid common vertical extent. The stack enumerates
these minima and their maximal spans; this covers the same row intersections
and contiguous Kp runs enumerated by the reference. Under the same supplied
Boolean matrix, Ki/Kp coordinates, `max_ki_gap` rules, positive-area
semantics, and tie-breaking order, the histogram formulation represents the
same valid axis-aligned rectangles as the reference's row-intersection
formulation. This equivalence is supported by the algorithmic reasoning
above and recorded reference-comparison tests; finite tests alone do not
prove universal correctness. It concerns rectangles in the sampled Boolean
matrix and does not establish stability between sampled points.

## 6. Complexity analysis

Let \(H\) be the number of supplied Ki rows, \(W\) the number of Kp samples,
and \(M\) the total number of interval entries across rows.

| Operation | Reference | Optimized |
|---|---:|---:|
| Rectangle search after matrix construction | \(O(H^2W)\) worst case | \(O(HW)\) |
| Boolean matrix memory | \(O(HW)\) | \(O(HW)\) |
| Additional search memory | \(O(W)\) | \(O(W)\) |

The reference examines \(O(H^2)\) top/bottom row pairs in the no-gap worst
case, with width-\(W\) row intersections and run detection for each pair.
The optimized search updates each of \(W\) bars for each of \(H\) rows; in
each row, every bar is pushed and popped at most once. Its search phase is
therefore \(O(HW)\), plus the existing \(O(W)\) histogram and stack state.

Matrix construction is separate from rectangle search. The implementation
allocates and initializes the Boolean matrix in \(O(HW)\). For each of the
\(M\) interval entries, it constructs a length-\(W\) coordinate mask and
paints the row, costing \(O(WM)\) in the implementation. Sorting supplied Ki
keys costs \(O(H\log H)\). Thus the full optimized function is

\[
O(H\log H + HW + WM),
\]

not unconditionally \(O(HW)\). It is \(O(HW)\) when the number of interval
entries per row is bounded, so \(M=O(H)\). The full reference function has
the same sorting and matrix-construction work plus its \(O(H^2W)\) search
phase.

## 7. Rectangle benchmarks

The kernel-only results in
[largest_rectangle_performance.csv](./benchmark_fair/largest_rectangle_performance.csv)
time the search kernels on an already-built matrix (10 repetitions).
The public-function results in
[largest_rectangle_end_to_end.csv](./benchmark_fair/largest_rectangle_end_to_end.csv)
include the function's matrix construction, use two warm-ups and 20 timed
repetitions per implementation, alternate execution order, and exclude input
generation and plotting. Each case used the same cached or fixed-seed
intervals for both implementations. All nine cases have exact matches for
`Kp_range`, `Ki_range`, `area`, and `center` (and matching `None` status).

| Case | Dimensions \(H\times W\) | Reference mean (ms) | Optimized mean (ms) | End-to-end speedup | Kernel-only speedup |
|---|---:|---:|---:|---:|---:|
| G1 cached | 49×300 | 18.386 | 16.007 | 1.15× | 1.13× |
| G5 cached | 18×300 | 2.539 | 4.578 | 0.55× | 0.52× |
| G6 cached | 15×300 | 1.731 | 3.933 | 0.44× | 0.44× |
| Synthetic | 32×64 | 6.899 | 2.117 | 3.26× | 3.29× |
| Synthetic | 64×128 | 28.290 | 8.489 | 3.33× | 3.57× |
| Synthetic | 128×256 | 113.117 | 32.941 | 3.43× | 3.93× |
| Synthetic | 128×512 | 123.008 | 64.292 | 1.91× | 2.07× |
| Synthetic | 256×256 | 454.557 | 63.885 | 7.12× | 7.60× |
| Synthetic | 256×512 | 493.359 | 131.489 | 3.75× | 3.81× |

For the cached plant maps, the optimized implementation is 1.15× faster on
G1, approximately 1.82× slower on G5 (the direct mean-time ratio is about
1.80×; 1.82× is the reciprocal of the rounded 0.55× speedup), and 2.27×
slower on G6. Its largest measured synthetic end-to-end speedup is 7.12×.

The original row-intersection implementation benefits from NumPy vectorized
AND, difference, and run-location operations. On small H, those vectorized
operations can be cheaper than the optimized implementation's Python-level
per-column histogram updates, stack operations, and candidate handling. This
explains why the optimized function is slower on the small cached G5 and G6
maps despite its better asymptotic search complexity.

As H grows, the reference repeatedly scans many top/bottom row combinations,
while the histogram implementation makes one linear pass per row. The
synthetic cases show this scaling benefit, reaching 7.12× end-to-end at
256×256. It is not a universal speedup: matrix construction is part of the
public API and can be a noticeable fraction of small cases, and the timing
results vary by input shape.

Kernel-only and end-to-end results use different measurement scopes and
different repetition counts. Kernel-only excludes matrix construction; the
public function includes it. Their speedups should be compared as separate
measurements, not treated as identical runs. The evidence does not justify
choosing a hybrid implementation threshold yet; a hybrid would require
additional crossover measurements and validation before implementation.

## 8. Correctness and validation record

The saved validation coverage includes:

- **Rectangle equivalence:** 16 focused parameterized/edge-case test
  invocations, including rectangular and irregular maps, disconnected
  intervals, nonuniform Ki spacing, below/equal/above gap cases, empty rows,
  boundary-touching regions, equal-area tie order, and degenerate inputs.
- **Randomized rectangle comparison:** 400 fixed-seed Boolean-map cases in
  `test_largest_stable_rectangle.py`, varying matrix dimensions, interval
  patterns, Ki spacing, and gap thresholds.
- **Search tests:** deterministic predicate tests for expansion, contraction,
  multiple intervals, disappearance, sentinels, fallback, periodic
  validation, global limits, and negative Ki direction.
- **Fair-harness tests:** shared lattice/endpoints, common absolute binary
  tolerance, and complete empty/terminating-row accounting.
- **Project test suite:** the recorded full-suite result is **29 passed**.
- **Syntax checks:** the recorded Python compilation and notebook code-cell
  syntax checks passed.
- **Rectangle benchmark:** nine cases, 20 timed repetitions per implementation
  after two warm-ups, with exact returned-field equivalence in every case.

Plant-level rectangle validation used actual cached nonempty interval
mappings for G1, G5, and G6. For G2, G3, and G4, the saved notebook logs show
no detected stable intervals; the reconstructed maps contain only the empty
seed row, `{0.0: []}`. Both functions return `None` for each of those empty
maps. This is not a comparison on nonempty G2–G4 stability maps.

The complete notebook was not executed from beginning to end. In particular,
saved outputs and individual benchmark harnesses do not establish that every
cell can be rerun in sequence with identical outputs.

## 9. End-to-end performance interpretation

In the recorded search runs, rectangle extraction was a small fraction of
search time. Across the rows in `multi_plant_runs.csv`, the measured
rectangle/search ratios range from approximately 0.010% to 0.086%, depending
on plant and strategy. This is consistent with the previously summarized
rough range of about 0.012–0.080%, with small endpoint differences when all
raw rows are included.

Even a large isolated rectangle speedup can therefore make little difference
to total stability-mapping runtime. For example, saving most of a rectangle
calculation that occupies less than one tenth of one percent of search time
still changes end-to-end time by less than that fraction. The repeated
simulation inside each classifier call is the more important performance
bottleneck; the fair G1 search takes tens of seconds, while rectangle
extraction takes roughly hundredths of a second.

## 10. Limitations and future work

1. **Finite numerical classifier:** response checks cover a finite simulation
   horizon and use numerical thresholds; they do not prove asymptotic
   stability.
2. **Finite spatial resolution:** all search policies can miss features
   between coarse Kp samples. Binary refinement improves only transitions
   that the sampled scan first observes.
3. **Adaptive continuity assumptions:** expanded-interval search can miss a
   new region outside the expanded prior intervals. Boundary-only can also
   reuse an interior containing an unseen narrow unstable region.
4. **Finite dense validation:** validation sampled 391 Kp points on five
   selected rows per plant. It is evidence for those coordinates, not a
   complete reference map.
5. **Stopping behavior:** once a direction has found a stable region, its
   first empty row terminates the search in that direction; a later
   reappearing stable region beyond that row is not explored.
6. **Sampled rectangle:** the rectangle optimizer works on the Boolean matrix
   constructed from interval results and sampled Kp coordinates. It does not
   classify every point in the continuous rectangle.
7. **Small-map overhead:** Python-level histogram/stack work can be slower
   than NumPy-vectorized reference work for small H.
8. **Incomplete G2–G4 rectangle evidence:** those plants have no independently
   saved nonempty interval maps, so no nonempty plant-level rectangle
   comparison is available for them.
9. **Notebook reproducibility:** the full notebook has not been executed
   top-to-bottom for this final validation.
10. **Hybrid not established:** current timings show a possible small-map
    crossover, but do not provide enough cases to select or validate a hybrid
    threshold.

Future work could test denser or adaptive reference sampling for features of
interest, record and compare full maps across more plants, investigate later
stable regions after an empty Ki row, and collect more rectangle timing cases
around a prospective hybrid crossover. These are recommendations, not
implemented changes.

## 11. Conclusions

The project has two distinct computational optimizations with different
evidence:

- **Stability search:** the fair benchmark uses a common Kp lattice and
  absolute refinement tolerance. Expanded-interval search delivered the best
  measured G1 call count and mean runtime among the tested strategies.
  Boundary-only reduced calls relative to exhaustive but was slower than
  expanded interval and failed the synthetic narrow-island validation; it
  should remain experimental rather than become the default.
- **Rectangle extraction:** the histogram/monotonic-stack search reduces the
  rectangle-search phase from \(O(H^2W)\) to \(O(HW)\), with matrix
  construction accounted for separately. Reference-equivalence tests and all
  nine end-to-end benchmark cases matched exactly. Its measured advantage
  depends on map size: the small cached G5/G6 maps were slower, while the
  larger synthetic maps were faster.

The practical priority for reducing current end-to-end runtime is classifier
and simulation work, not rectangle extraction. The results support the
expanded-interval search as the best tested practical search policy and the
optimized rectangle routine as a scalable option for larger maps, while
retaining the stated finite-resolution and reproducibility limitations.

## Sources and evidence

Primary implementation and tests:

- [stability_without_poles.ipynb](./stability_without_poles.ipynb) — plant
  definitions, simulation classifier, search invocation, rectangle reference,
  plotting, and current optimized-function import.
- [stability_without_poles_original_backup.ipynb](./stability_without_poles_original_backup.ipynb)
  — preserved pre-optimization notebook.
- [stability_search.py](./stability_search.py) — the three search policies,
  interval scanning, boundary refinement, sentinels, fallbacks, and stats.
- [largest_stable_rectangle.py](./largest_stable_rectangle.py) — optimized
  coordinate-aware histogram/stack implementation.
- [test_largest_stable_rectangle.py](./test_largest_stable_rectangle.py) and
  [test_stability_search.py](./test_stability_search.py) — equivalence and
  search tests.

Saved benchmark reports and data:

- [stability_fair_benchmark_report.md](./stability_fair_benchmark_report.md)
  and [stability_strategy_report.md](./stability_strategy_report.md) — prior
  search benchmark summaries. The latter is a historical report; the current
  notebook's G1 display label has since been corrected to match its transfer
  function.
- [original_implementation_benchmark.csv](./benchmark_fair/original_implementation_benchmark.csv),
  [repeated_timing_summary.csv](./benchmark_fair/repeated_timing_summary.csv),
  [repeated_run_timings.csv](./benchmark_fair/repeated_run_timings.csv),
  [multi_plant_runs.csv](./benchmark_fair/multi_plant_runs.csv), and
  [multi_plant_summary.csv](./benchmark_fair/multi_plant_summary.csv).
- [dense_grid_validation.csv](./benchmark_fair/dense_grid_validation.csv),
  [sentinel_density_comparison.csv](./benchmark_fair/sentinel_density_comparison.csv),
  [narrow_island_sentinel_test.csv](./benchmark_fair/narrow_island_sentinel_test.csv),
  [ki_row_accounting.csv](./benchmark_fair/ki_row_accounting.csv), and
  [common_ki_rows.csv](./benchmark_fair/common_ki_rows.csv).
- [largest_rectangle_performance.csv](./benchmark_fair/largest_rectangle_performance.csv)
  and [largest_rectangle_end_to_end.csv](./benchmark_fair/largest_rectangle_end_to_end.csv),
  with their runners
  [benchmark_largest_rectangle.py](./benchmark_fair/benchmark_largest_rectangle.py)
  and
  [benchmark_largest_rectangle_end_to_end.py](./benchmark_fair/benchmark_largest_rectangle_end_to_end.py).

All numerical benchmark values in this report were checked against the
corresponding saved CSVs and reports. The unavailable evidence is nonempty
saved G2–G4 interval mappings and a full top-to-bottom notebook execution.
No implementation, notebook, experimental result, or existing benchmark
artifact was changed to produce this report.

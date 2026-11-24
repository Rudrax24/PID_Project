# PID Controller Stability Mapping

This project estimates closed-loop stability regions in the proportional-integral
gain plane ($K_p$, $K_i$) for six transfer-function plants. It uses a
simulation-based classifier and adaptive searches to map sampled regions, then
compares the largest stable rectangle found in each map. The results are
numerical evidence for the configured simulation and sampling settings, not a
mathematical proof of stability.

## Project Objectives

- Classify closed-loop responses without deriving stability boundaries from
  pole locations.
- Compare exhaustive, expanded-interval, and boundary-only searches.
- Find the largest axis-aligned stable rectangle in a sampled $K_p$-$K_i$
  map.
- Evaluate search and rectangle implementations for correctness and performance.

## Methodology

### Simulation-based stability classification

The main notebook defines six transfer-function plants and uses a finite-horizon
closed-loop step-response simulation to classify each tested $(K_p, K_i)$
point. The classifier checks the simulated response against numerical,
settling, and output-bound criteria. A point's classification therefore depends
on the simulation horizon, time step, thresholds, and numerical behavior; it
does not establish asymptotic stability.

### Stability-search strategies

The adaptive search in [`stability_search.py`](stability_search.py) supports:

- **`exhaustive`** — scans the full $K_p$ range on each visited $K_i$ row.
- **`expanded_interval`** — scans the previous row's detected stable
  intervals, expanded by a configured margin. This is the practical baseline
  used in the comparisons.
- **`boundary_only`** — samples neighborhoods around previous interval
  boundaries and probes selected interior sentinels, reusing the remaining
  interior when checks pass. Fallback and periodic full-row scans catch some
  discrepancies, but finite probes can miss narrow features. Treat this
  strategy as experimental.

### Largest stable rectangle

[`largest_stable_rectangle.py`](largest_stable_rectangle.py) constructs a
sampled Boolean stability map and searches it using coordinate-aware histogram
heights and a monotonic stack. Its rectangle-search stage is $O(HW)$ for a
map with $H$ $K_i$ rows and $W$ $K_p$ samples, after the map is built;
the reference search is $O(H^2W)$ after map construction. Matrix creation,
interval painting, and Python-level processing also contribute to full-function
runtime, so the lower search complexity does not make the optimized function
faster for every input size.

The original rectangle implementation is retained in the main notebook for
equivalence testing. The tests compare the optimized function's returned
rectangle fields with the reference across structured, edge, and deterministic
randomized inputs.

### Fair benchmarking

The fair search benchmark uses a shared full-range $K_p$ lattice and a common
absolute boundary-refinement tolerance so the strategies are compared at
matched spatial resolution. It also records repeated timings, classifier-call
categories, row accounting, and finite dense-grid validation. This fair
experiment is separate from the earlier root-level, single-run benchmark
artifacts; their measurements should not be combined.

## Repository Structure

```text
.
├── stability_without_poles.ipynb       # Main six-plant workflow
├── stability_search.py                 # Adaptive search strategies
├── largest_stable_rectangle.py         # Optimized rectangle search
├── test_stability_search.py            # Search tests
├── test_largest_stable_rectangle.py    # Rectangle/reference tests
├── benchmark_fair/
│   ├── fair_benchmark.py               # Resolution-matched harness
│   ├── run_fair_benchmarks.py          # Search benchmark runner
│   ├── benchmark_largest_rectangle.py
│   ├── benchmark_largest_rectangle_end_to_end.py
│   ├── test_fair_benchmark.py
│   ├── notebook_strategy_comparison/   # Separate notebook-cell execution
│   └── *.csv, *.png                    # Recorded benchmark evidence
├── FINAL_TECHNICAL_REPORT.md
├── stability_fair_benchmark_report.md
├── stability_strategy_report.md
└── [legacy notebooks, scripts, and results]
```

The notebooks and scripts `kp_vs_ki_plot.ipynb`, `pid_controller.ipynb`,
`plots_and_stability.ipynb`, `ray_trace_alg.ipynb`, `new.ipynb`,
`stability_without_poles_original_backup.ipynb`, `compare_plot.py`, and
`first_order.py` are historical material, not part of the primary run-and-test
workflow.

The four legacy PID parameter-result files
(`pid_parameter_results.csv`, `first_order_pid_parameter_results.csv`,
`second_order_pid_parameter_results.csv`, and
`third_order_pid_parameter_results.csv`) are byte-identical in this repository.
They should not be treated as four independent plant datasets.

## Requirements and Setup

The fair benchmark uses the Python 3.10 union type syntax (`tuple | None`), so
use **Python 3.10 or later**. The project imports these direct Python packages:

- `numpy`
- `control` (the `python-control` package)
- `matplotlib`
- `pandas`
- `pytest` (for tests)

No dependency manifest or pinned package versions are provided. Create an
environment and install the core project packages with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy control matplotlib pandas pytest
```

Notebook frontend packages are optional and are not core project dependencies.
Choose an option if you want to run the notebook:

- **Jupyter Notebook/JupyterLab:** `python -m pip install jupyter`
- **VS Code notebook support:** `python -m pip install ipykernel` in the
  selected environment. VS Code's Python and Jupyter extensions are also
  required.

These installation commands are based on inspected imports; a clean environment
installation has not been verified.

## How to Run

Run commands from the repository root. In particular, the optional strategy
comparison cell writes to a path relative to the current working directory.

1. Open [`stability_without_poles.ipynb`](stability_without_poles.ipynb) in
   Jupyter or VS Code.
2. In a clean kernel, execute cells in notebook order: imports and plant
   definitions, the simulation classifier, rectangle definitions/import,
   adaptive-search and plotting definitions, and the six-plant mapping.
3. The final strategy-comparison cell is optional. It runs the comparison and
   writes four files under
   `benchmark_fair/notebook_strategy_comparison/`.

The complete notebook has not been verified in a clean-kernel, top-to-bottom
run. Re-running the strategy-comparison cell overwrites its four CSV/PNG
outputs. The benchmark runner scripts also write their recorded outputs under
`benchmark_fair/`; running them can overwrite existing evidence and may take
substantial time.

Run the test suite from the repository root:

```powershell
python -m pytest -q -p no:cacheprovider
```

To deliberately reproduce the fair search benchmark, run
`python benchmark_fair/run_fair_benchmarks.py`. The rectangle benchmark scripts
are separate: `python benchmark_fair/benchmark_largest_rectangle.py` and
`python benchmark_fair/benchmark_largest_rectangle_end_to_end.py`. These
commands regenerate benchmark outputs; do not run them when preserving the
checked-in measurements is important.

## Benchmark Results

### Stability-search strategies

The fair, five-run G1 comparison and the earlier root-level single-run results
are separate experiments:

| Experiment | Strategy | Mean search time | Classifier calls |
|---|---|---:|---:|
| Fair repeated G1 | Exhaustive | 27.491 s | 8,011 |
| Fair repeated G1 | Expanded interval | 23.484 s | 6,942 |
| Fair repeated G1 | Boundary-only | 25.126 s | 7,448 |
| Earlier single run | Exhaustive | 27.002 s | 8,058 |
| Earlier single run | Expanded interval | 26.855 s | 8,058 |
| Earlier single run | Boundary-only | 25.337 s | 7,434 |

In the fair G1 experiment, expanded-interval had the lowest mean time and call
count. Boundary-only used fewer calls and less time than exhaustive in that
sample, but more calls and more time than expanded-interval. The earlier
single-run results are not resolution-matched and are shown separately rather
than treated as confirmation of the fair results. Boundary-only also missed a
narrow synthetic interior instability between its probes, demonstrating a
known correctness risk.

### Largest-rectangle implementation

The optimized function exactly matched the reference result fields in all nine
recorded benchmark cases, but was not universally faster:

- Cached G1 map: **1.15× faster**.
- Cached G5 map: approximately **1.80× slower**.
- Cached G6 map: **2.27× slower**.
- The largest measured synthetic end-to-end speedup was **7.12×**.

The reports and CSVs contain the detailed measurements and specify whether
timings include matrix construction. These results are specific to the tested
maps and machine; the smaller cached maps show why asymptotic complexity alone
does not predict runtime.

## Limitations

- The classifier uses finite-horizon numerical simulation; it is not a
  mathematical proof of asymptotic stability.
- The search samples finite $K_p$ and $K_i$ coordinates. Dense validation
  covers selected rows only and cannot prove that all missed regions or
  boundaries have been found.
- Boundary-only search can miss narrow interior features between sentinels.
  Its fallback and periodic scans are also finite-resolution checks and do not
  eliminate that risk.
- The saved mapping reports no detected stable intervals for G2, G3, and G4
  under the configured search. This is not proof that no stable region exists.
- Runtime measurements vary with system load and input size. The optimized
  rectangle implementation is slower on the recorded cached G5 and G6 maps.
- The full notebook has not been verified from a clean kernel, top to bottom.
- Numerical warnings were observed for ill-conditioned or diverging trials in
  the simulation stack.

## Reports and Further Reading

- [Final Technical Report](FINAL_TECHNICAL_REPORT.md)
- [Fair Benchmark and Validation Report](stability_fair_benchmark_report.md)
- [Strategy Comparison and Complexity Report](stability_strategy_report.md)

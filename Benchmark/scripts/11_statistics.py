#!/usr/bin/env python3
"""Phase 7b: results/statistics.csv.

PLAN.md Section 10.3. Reads only `results/designs.csv` and
`results/scaffold_summary.csv`, both written by `10_aggregate.py`.

**The unit of analysis is the scaffold x target cell, never the design.** Ten
designs from one scaffold are not ten independent observations, and Section 10.3
is explicit that aggregation happens before testing. Every test below pairs
`mpnn_soluble` against one other arm on cells that share a scaffold and a charge
target, so each scaffold contributes one value per method per target.

**Effect sizes are reported for every test**, per Section 10.3: the median paired
difference with a BCa bootstrap 95% CI over 10,000 resamples of the pairs, and
Cliff's delta. A p-value alone is not an acceptable output here.

McNemar and the Wilson interval are implemented in this file rather than taken
from statsmodels, which is not in `py311`. Exact McNemar is a two-sided binomial
test on the discordant pairs, so it needs nothing beyond `scipy.stats.binomtest`,
and adding a dependency to a shared environment for one function would be a
poor trade. Both are checked against worked examples in `--self-test`.

**A confound that the hit-rate comparison cannot remove.** The arms do not have
equal sampling budgets: `avnapsa` produces one design per cell, every other arm
produces ten. `hit_any` therefore rewards arms that get ten attempts. It is
reported because Section 10.3 asks for a paired per-cell exact-hit comparison,
alongside `hit_rate`, the per-cell fraction of designs on target, which is not
budget-inflated in the same way. Neither is a substitute for the other and the
`metric` column distinguishes them.

Usage:
  11_statistics.py [--dry-run] [--self-test] [--bootstrap N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402

STATISTICS_COLUMNS = ["metric", "comparison", "stratum", "n_pairs", "test",
                      "statistic", "p_raw", "p_holm", "median_diff",
                      "ci_low", "ci_high", "cliffs_delta"]

REFERENCE_ARM = "mpnn_soluble"

# Continuous metrics named in Section 10.3, plus the Section 10.2 diversity
# metrics it folds into the same treatment. Column names are scaffold_summary's.
CONTINUOUS_METRICS = [
    "hit_rate", "median_mut_per_charge", "median_d_reu_per_res",
    "median_d_hbond_strong", "median_d_plddt_vs_wt_pred",
    "median_tm_score_to_wt_crystal",
    "n_unique_mutation_sets", "mean_pairwise_hamming", "positional_entropy",
    "designable_coverage",
]

N_BOOTSTRAP = 10_000


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# statistics implemented here
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at the rates this benchmark
    produces: several arms sit at or very near 0 or 1, where the Wald interval
    is degenerate or runs outside [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Exact two-sided McNemar on the discordant counts.

    `b` is the number of pairs where the reference succeeded and the comparator
    did not, `c` the reverse. Concordant pairs carry no information about a
    difference and are excluded by construction. With b + c == 0 there is
    nothing to test and the p-value is 1.0.
    """
    from scipy.stats import binomtest                              # noqa: PLC0415

    n = b + c
    if n == 0:
        return (float("nan"), 1.0)
    return (float(b), float(binomtest(b, n, 0.5, alternative="two-sided").pvalue))


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's delta, the probability of dominance, in [-1, 1].

    Computed on the two samples rather than on the paired differences, because
    it is a measure of overlap between distributions. Ties count as neither
    greater nor less, which is the standard definition.
    """
    if not x or not y:
        return float("nan")
    greater = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return (greater - less) / (len(x) * len(y))


def bca_median_diff_ci(diffs: list[float], n_resamples: int) -> tuple[float, float]:
    """BCa bootstrap 95% CI for the median paired difference."""
    import numpy as np                                             # noqa: PLC0415
    from scipy.stats import bootstrap                              # noqa: PLC0415

    arr = np.asarray(diffs, dtype=float)
    if len(arr) < 3 or np.allclose(arr, arr[0]):
        # BCa needs variation and enough pairs to jackknife. A degenerate sample
        # gets an empty interval rather than a fabricated one.
        return (float("nan"), float("nan"))
    try:
        res = bootstrap((arr,), np.median, n_resamples=n_resamples,
                        confidence_level=0.95, method="BCa",
                        random_state=np.random.default_rng(20260813))
        return (float(res.confidence_interval.low),
                float(res.confidence_interval.high))
    except Exception:                                              # noqa: BLE001
        return (float("nan"), float("nan"))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    indexed = sorted((p, i) for i, p in enumerate(pvals) if p == p)
    out = [float("nan")] * len(pvals)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)     # enforce monotonicity
        out[i] = running
    return out


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------

def run(cfg: dict, dry_run: bool, n_resamples: int) -> None:
    import pandas as pd                                            # noqa: PLC0415
    from scipy.stats import wilcoxon                               # noqa: PLC0415

    summary_path = config_lib.bench_path("results", "scaffold_summary.csv")
    out_path = config_lib.bench_path("results", "statistics.csv")

    if dry_run:
        print(f"[dry-run] would read {summary_path} "
              f"({'present' if summary_path.exists() else 'NOT PRESENT'})")
        print(f"[dry-run] would pair every arm against {REFERENCE_ARM} on "
              "(scaffold_id, target_charge) cells")
        print(f"[dry-run] would run exact McNemar on hit_any and Wilcoxon "
              f"signed-rank on {len(CONTINUOUS_METRICS)} continuous metrics, "
              "overall and per delta_q_density")
        print(f"[dry-run] would report BCa CIs from {n_resamples} resamples and "
              "Cliff's delta for every test, Holm-corrected within each metric")
        print(f"[dry-run] would write {out_path} with columns {STATISTICS_COLUMNS}")
        return

    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}. Run 10_aggregate.py first.")
    df = pd.read_csv(summary_path)
    methods = [m for m in sorted(df["method"].unique()) if m != REFERENCE_ARM]
    if REFERENCE_ARM not in set(df["method"]):
        raise SystemExit(f"{REFERENCE_ARM} is absent from {summary_path}")

    # hit_any: did the cell produce at least one on-target design.
    df["hit_any"] = (df["n_hit_exact"].fillna(0) > 0).astype(int)

    strata = [("all", df)]
    for density in sorted(df["delta_q_density"].unique()):
        strata.append((f"delta_q_density={density:+d}",
                       df[df["delta_q_density"] == density]))

    rows: list[dict] = []
    for method in methods:
        for stratum_name, sub in strata:
            ref = sub[sub["method"] == REFERENCE_ARM]
            cmp_ = sub[sub["method"] == method]
            merged = ref.merge(cmp_, on=["scaffold_id", "target_charge"],
                               suffixes=("_ref", "_cmp"))
            if merged.empty:
                continue

            # ---- binary exact-hit, McNemar ------------------------------
            b = int(((merged["hit_any_ref"] == 1) & (merged["hit_any_cmp"] == 0)).sum())
            c = int(((merged["hit_any_ref"] == 0) & (merged["hit_any_cmp"] == 1)).sum())
            stat, p = mcnemar_exact(b, c)
            n_pairs = len(merged)
            ref_rate = merged["hit_any_ref"].mean()
            cmp_rate = merged["hit_any_cmp"].mean()
            lo_r, hi_r = wilson_interval(int(merged["hit_any_ref"].sum()), n_pairs)
            lo_c, hi_c = wilson_interval(int(merged["hit_any_cmp"].sum()), n_pairs)
            rows.append({
                "metric": "hit_any", "comparison": f"{REFERENCE_ARM}_vs_{method}",
                "stratum": stratum_name, "n_pairs": n_pairs,
                "test": f"mcnemar_exact(b={b},c={c}); wilson ref "
                        f"[{lo_r:.3f},{hi_r:.3f}] cmp [{lo_c:.3f},{hi_c:.3f}]",
                "statistic": stat, "p_raw": p, "p_holm": "",
                "median_diff": ref_rate - cmp_rate,
                "ci_low": "", "ci_high": "",
                "cliffs_delta": cliffs_delta(list(merged["hit_any_ref"]),
                                             list(merged["hit_any_cmp"])),
            })

            # ---- continuous metrics, Wilcoxon signed-rank ----------------
            for metric in CONTINUOUS_METRICS:
                a, bcol = f"{metric}_ref", f"{metric}_cmp"
                if a not in merged or bcol not in merged:
                    continue
                pair = merged[[a, bcol]].dropna()
                if len(pair) < 3:
                    continue
                x = pair[a].astype(float).tolist()
                y = pair[bcol].astype(float).tolist()
                diffs = [u - v for u, v in zip(x, y)]
                if all(d == 0 for d in diffs):
                    stat_w, p_w = float("nan"), 1.0
                else:
                    try:
                        res = wilcoxon(x, y, zero_method="wilcox",
                                       alternative="two-sided")
                        stat_w, p_w = float(res.statistic), float(res.pvalue)
                    except ValueError:
                        stat_w, p_w = float("nan"), float("nan")
                lo, hi = bca_median_diff_ci(diffs, n_resamples)
                med = sorted(diffs)[len(diffs) // 2] if len(diffs) % 2 else \
                    (sorted(diffs)[len(diffs) // 2 - 1] + sorted(diffs)[len(diffs) // 2]) / 2
                rows.append({
                    "metric": metric,
                    "comparison": f"{REFERENCE_ARM}_vs_{method}",
                    "stratum": stratum_name, "n_pairs": len(pair),
                    "test": "wilcoxon_signed_rank",
                    "statistic": stat_w, "p_raw": p_w, "p_holm": "",
                    "median_diff": med, "ci_low": lo, "ci_high": hi,
                    "cliffs_delta": cliffs_delta(x, y),
                })

    # Holm within each metric family: all (comparison, stratum) tests of a metric.
    by_metric: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_metric.setdefault(row["metric"], []).append(i)
    for metric, idx in by_metric.items():
        adjusted = holm([rows[i]["p_raw"] for i in idx])
        for i, p in zip(idx, adjusted):
            rows[i]["p_holm"] = p

    bio.write_csv(out_path, rows, STATISTICS_COLUMNS)
    log(f"wrote {out_path} with {len(rows)} tests")
    log(f"  {len(methods)} comparisons against {REFERENCE_ARM}, "
        f"{len(strata)} strata, {len(by_metric)} metrics")
    sig = sum(1 for r in rows if isinstance(r["p_holm"], float) and r["p_holm"] < 0.05)
    log(f"  {sig} tests significant at Holm-corrected p < 0.05")
    log("  Fold-class effect: per-class descriptive statistics only, no linear "
        "mixed model. Section 10.3 permits this and requires saying which was done.")


def self_test() -> int:
    """Check the hand-rolled statistics against worked values."""
    ok = True

    lo, hi = wilson_interval(9, 10)
    if not (0.55 < lo < 0.60 and 0.98 < hi < 1.0):
        print(f"FAIL wilson_interval(9,10) = ({lo:.4f}, {hi:.4f})")
        ok = False
    lo, hi = wilson_interval(0, 10)
    if not (lo == 0.0 and 0.27 < hi < 0.29):
        print(f"FAIL wilson_interval(0,10) = ({lo:.4f}, {hi:.4f})")
        ok = False

    # Exact McNemar with b=10, c=2 is a binomial test of 10/12 at p=0.5.
    _, p = mcnemar_exact(10, 2)
    if not (0.038 < p < 0.040):
        print(f"FAIL mcnemar_exact(10,2) p = {p:.5f}")
        ok = False
    _, p = mcnemar_exact(0, 0)
    if p != 1.0:
        print(f"FAIL mcnemar_exact(0,0) p = {p}")
        ok = False

    # Cliff's delta: complete dominance is 1, identical samples 0.
    if cliffs_delta([4, 5, 6], [1, 2, 3]) != 1.0:
        print("FAIL cliffs_delta complete dominance")
        ok = False
    if cliffs_delta([1, 2, 3], [1, 2, 3]) != 0.0:
        print("FAIL cliffs_delta identical")
        ok = False

    # Holm: monotone, and the smallest p multiplied by m.
    adj = holm([0.01, 0.04, 0.03])
    if not (abs(adj[0] - 0.03) < 1e-12 and adj[1] >= adj[2] >= adj[0]):
        print(f"FAIL holm = {adj}")
        ok = False

    print("self-test: " + ("all checks passed" if ok else "FAILURES above"))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    run(config_lib.load_benchmark(), args.dry_run, args.bootstrap)


if __name__ == "__main__":
    main()

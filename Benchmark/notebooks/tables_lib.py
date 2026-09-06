"""Table builders T1 to T6 for notebooks/analysis.ipynb.

PLAN.md Section 11.2. Same rules as `analysis_lib`: reads only `results/*.csv`
and `data/scaffold_manifest.csv`, writes only `tables/`, and never recomputes a
metric that `10_aggregate.py` or `11_statistics.py` already wrote.

T1 to T5 are exported as LaTeX ready to paste into the manuscript; T6 is the full
statistics table and is supplementary.
"""

from __future__ import annotations

import pandas as pd

import analysis_lib as A
from analysis_lib import ROOT, write_table
# Arm identity comes from figures_lib, which is where the nine-arm order,
# labels and palette are defined. Importing the six-arm pair that used to
# live in analysis_lib is what silently dropped the method label from every
# mpnn_hyper, mpnn_halo and rosetta_hbond_off row. See _order_methods.
from figures_lib import METHOD_LABEL, METHOD_ORDER


def _order_methods(df: pd.DataFrame, col: str = "method") -> pd.DataFrame:
    """Sort by the frozen arm order, keeping any arm the order does not name.

    A plain `Categorical(categories=METHOD_ORDER)` maps anything outside the
    list to NaN, so an arm added to `designs.csv` but not to METHOD_ORDER keeps
    its row and loses its name. That is how 42 rows of T7 came to have a blank
    method column. Unknown arms now sort to the end under their own name, and
    the caller is told, so a new arm is visible rather than anonymous.
    """
    df = df.copy()
    unknown = sorted(set(df[col].dropna().unique()) - set(METHOD_ORDER))
    if unknown:
        print(f"note: {col} values not in METHOD_ORDER, sorted last: {unknown}")
    df[col] = pd.Categorical(df[col], categories=list(METHOD_ORDER) + unknown,
                             ordered=True)
    return df.sort_values(col)


def table1(data: dict) -> pd.DataFrame:
    """Scaffold panel."""
    m = data["manifest"].copy()
    cols = ["scaffold_id", "pdb_id", "chain", "n_residues", "resolution",
            "fold_class", "frac_helix", "frac_strand", "frac_coil",
            "q_wt", "q_wt_per100", "n_designable", "source_split"]
    t = m[[c for c in cols if c in m.columns]].sort_values(
        ["fold_class", "scaffold_id"])
    write_table(t, "T1_scaffold_panel",
                "Scaffold panel: source, size, fold class, secondary structure "
                "fractions, wild-type net charge and designable positions.")
    return t


def table2(data: dict) -> pd.DataFrame:
    """Headline paired comparison, with effect sizes from statistics.csv."""
    s = data["summary"]
    d = data["designs"]
    rows = []
    for method in [m for m in METHOD_ORDER if m in set(s["method"])]:
        sm = s[s["method"] == method]
        dm = d[d["method"] == method]
        rows.append({
            "method": METHOD_LABEL[method],
            "n_cells": len(sm),
            "n_designs": len(dm),
            "hit_rate": dm["hit_exact"].mean(),
            "mut_per_charge_median": dm["mut_per_charge"].median(),
            "unique_sets_median": sm["n_unique_mutation_sets"].median(),
            "pairwise_hamming_median": sm["mean_pairwise_hamming"].median(),
            "d_reu_per_res_median": dm["d_reu_per_res"].median(),
            "d_hbond_strong_median": dm["d_hbond_strong"].median()
            if "d_hbond_strong" in dm else float("nan"),
            "d_plddt_vs_wt_median": dm["d_plddt_vs_wt_pred"].median()
            if "d_plddt_vs_wt_pred" in dm else float("nan"),
        })
    t = pd.DataFrame(rows)

    stats = data.get("stats")
    if stats is not None and len(stats):
        key = stats[(stats["stratum"] == "all")
                    & (stats["metric"].isin(["hit_rate", "median_mut_per_charge"]))]
        extra = []
        for _, row in key.iterrows():
            extra.append({
                "metric": row["metric"], "comparison": row["comparison"],
                "median_diff": row["median_diff"], "ci_low": row["ci_low"],
                "ci_high": row["ci_high"], "cliffs_delta": row["cliffs_delta"],
                "p_holm": row["p_holm"],
            })
        if extra:
            write_table(pd.DataFrame(extra), "T2b_effect_sizes",
                        "Paired effect sizes against mpnn_soluble, all strata "
                        "pooled: median paired difference with BCa 95% CI, "
                        "Cliff's delta and Holm-corrected p.")
    write_table(t, "T2_headline_comparison",
                "Headline comparison by method. Hit rate is per design; "
                "diversity is per scaffold x target cell.")
    return t


def table3(data: dict) -> pd.DataFrame:
    """The eGFP table from the manuscript, extended with prediction columns."""
    d = data["designs"]
    e = d[d["scaffold_id"] == "eGFP"]
    if e.empty:
        print("note: no eGFP rows, T3 skipped")
        return pd.DataFrame()
    t = (e.groupby(["method", "delta_q_density", "target_charge"])
         .agg(n_designs=("design_id", "size"),
              hit_rate=("hit_exact", "mean"),
              q_actual_median=("q_actual", "median"),
              n_mutations_median=("n_mutations", "median"),
              mut_per_charge_median=("mut_per_charge", "median"),
              d_reu_per_res_median=("d_reu_per_res", "median"),
              plddt_mean=("plddt_mean", "mean"),
              ca_rmsd=("ca_rmsd_to_wt_crystal", "median"),
              tm_score=("tm_score_to_wt_crystal", "median"))
         .reset_index())
    t = _order_methods(t)
    write_table(t, "T3_egfp_extended",
                "eGFP focus scaffold, every arm and every charge target, "
                "extended with ESMFold pLDDT, CA RMSD and TM-score to the "
                "wild-type crystal.")
    return t


def table4(data: dict) -> pd.DataFrame:
    """AF3 subset."""
    af3 = data.get("af3")
    if af3 is None or af3.empty:
        print("note: results/af3.csv absent or empty, T4 skipped")
        return pd.DataFrame()
    cols = ["design_id", "scaffold_id", "method", "delta_q_density",
            "target_charge", "af3_plddt", "af3_ptm", "af3_ca_rmsd",
            "af3_tm_score", "af3_msa_mode", "status"]
    t = af3[[c for c in cols if c in af3.columns]].copy()
    write_table(t, "T4_af3_subset",
                "AlphaFold3 stratified subset, single-sequence mode. RMSD is to "
                "the wild-type crystal chain, not to the threaded model.")
    return t


def table5(data: dict) -> pd.DataFrame:
    """Runtime and computational overhead, including the rejection alternative."""
    d = data["designs"]
    per_cell = (d.groupby(["method", "scaffold_id", "target_charge"])
                .agg(wall=("wall_seconds", "first"),
                     n=("design_id", "size"),
                     hits=("hit_exact", "sum")).reset_index())
    rows = []
    for method in [m for m in METHOD_ORDER if m in set(per_cell["method"])]:
        pm = per_cell[per_cell["method"] == method]
        with_hits = pm[pm["hits"] > 0]
        rows.append({
            "method": METHOD_LABEL[method],
            "n_cells": len(pm),
            "cell_seconds_median": pm["wall"].median(),
            "cell_seconds_total_h": pm["wall"].sum() / 3600.0,
            "cells_with_no_hit": int((pm["hits"] == 0).sum()),
            "seconds_per_on_target_median":
                (with_hits["wall"] / with_hits["hits"]).median()
                if len(with_hits) else float("nan"),
        })
    t = pd.DataFrame(rows)

    curve = data.get("curve")
    dist = data.get("dist")
    if curve is not None and len(curve) and dist is not None and len(dist):
        censored = curve["expected_gpu_seconds_per_hit"].astype(str).str.startswith(">")
        known = curve[~censored]
        # One pool per scaffold serves all 8 of that scaffold's targets, so the
        # total cost comes from the per-scaffold distribution table. Summing the
        # curve instead would count each pool once per target, an 8x overstatement.
        pool_seconds = (dist["seconds_per_sample"].astype(float)
                        * dist["n_samples_drawn"].astype(float))
        t = pd.concat([t, pd.DataFrame([{
            "method": "Vanilla ProteinMPNN + rejection",
            "n_cells": len(curve),
            "cell_seconds_median": pool_seconds.median(),
            "cell_seconds_total_h": pool_seconds.sum() / 3600.0,
            "cells_with_no_hit": int(censored.sum()),
            "seconds_per_on_target_median":
                known["expected_gpu_seconds_per_hit"].astype(float).median()
                if len(known) else float("nan"),
        }])], ignore_index=True)
    write_table(t, "T5_runtime",
                "Runtime and computational overhead. All values are CPU seconds "
                "under Decision D. The rejection row's cost per on-target design "
                "covers only the cells where rejection produced a hit; the "
                "cells_with_no_hit column counts the cells where it produced "
                "none and no ratio exists.")
    return t


def table6(data: dict) -> pd.DataFrame:
    """Full statistics table, supplementary."""
    stats = data.get("stats")
    if stats is None or stats.empty:
        print("note: results/statistics.csv absent, T6 skipped")
        return pd.DataFrame()
    write_table(stats, "T6_statistics_full",
                "Full statistical treatment: exact McNemar on per-cell hit, "
                "Wilcoxon signed-rank on continuous metrics paired by scaffold "
                "and target, Holm-corrected within each metric, with BCa 95% "
                "confidence intervals and Cliff's delta.")
    return stats


def table7(data: dict) -> pd.DataFrame:
    """T7 and T7b, PLAN.md Section 11.2, added 2026-08-26.

    The question this answers was posed as "at what temperature or mutation
    count does pLDDT drop below 0.8". Two things stop that being answerable as
    asked, and both are carried in the table rather than hidden:

    1. 10 of the 25 wild types already predict below 80, so `frac_below_abs80`
       is context and not a result. Every ΔpLDDT column is against the
       scaffold's own predicted wild type instead.
    2. The mutation-count axis is unbalanced across scaffolds: only the longest
       scaffolds reach the high bins. Every summary is therefore the median of
       per-scaffold medians, with the spread taken across scaffolds, and bins
       holding fewer than `A.MIN_SCAFFOLDS_PER_BIN` scaffolds are dropped.
       `n_scaffolds` and `n_designs` are both reported so the imbalance stays
       visible.

    T7b reduces T7 to the first bin on each axis where the balanced median
    crosses each level. A method that never crosses gets "none", not a number.
    """
    d = A.plddt_frame(data)
    if d.empty:
        return pd.DataFrame()

    axes = [("mutations", "mut_bin"),
            ("abs_delta_q_density", "abs_density"),
            ("final_temperature", "final_temperature")]

    rows = []
    for axis_name, col in axes:
        frame = d
        if axis_name == "final_temperature":
            # Only the MPNN arms decode with a temperature; the others have no
            # value on this axis and are left out rather than binned at NaN.
            frame = d[d["method"].str.startswith("mpnn")]
            frame = frame.dropna(subset=["final_temperature"])
        for method, m in frame.groupby("method"):
            bal = A.scaffold_balanced(m, col, "d_plddt_vs_wt_pred")
            for key, b in bal.iterrows():
                sub = m[m[col] == key]
                delta = sub["d_plddt_vs_wt_pred"]
                row = {
                    "method": method, "axis": axis_name, "bin": str(key),
                    "n_designs": int(b["n_designs"]),
                    "n_scaffolds": int(b["n_scaffolds"]),
                    "median_d_plddt": round(float(b["median"]), 2),
                    "q25_across_scaffolds": round(float(b["q25"]), 2),
                    "q75_across_scaffolds": round(float(b["q75"]), 2),
                    "pooled_median_d_plddt": round(float(delta.median()), 2),
                    "median_plddt_abs": round(float(sub["plddt_mean"].median()), 2),
                    "frac_below_abs80": round(float(sub["below_abs"].mean()), 3),
                }
                for level in A.PLDDT_DROP_LEVELS:
                    row[f"frac_below_{int(abs(level))}"] = round(
                        float((delta < level).mean()), 3)
                rows.append(row)
    t = _order_methods(pd.DataFrame(rows))
    t = t.sort_values(["method", "axis", "n_designs"], kind="stable")

    # T7b: first bin on each axis whose balanced median crosses each level.
    cross = []
    for (method, axis_name), sub in t.groupby(["method", "axis"], observed=True):
        if axis_name == "mutations":
            keys = [b for b in A.MUT_BIN_LABELS if b in set(sub["bin"])]
        else:
            keys = sorted(set(sub["bin"]), key=float)
        ordered = sub.set_index("bin").loc[keys]
        for level in A.PLDDT_DROP_LEVELS:
            hit = ordered[ordered["median_d_plddt"] < level]
            cross.append({
                "method": method, "axis": axis_name,
                "level_d_plddt": level,
                "first_bin_crossing": hit.index[0] if len(hit) else "none",
                "n_designs_at_crossing": int(hit["n_designs"].iloc[0]) if len(hit) else 0,
                "n_scaffolds_at_crossing": int(hit["n_scaffolds"].iloc[0]) if len(hit) else 0,
                "median_at_crossing": (float(hit["median_d_plddt"].iloc[0])
                                       if len(hit) else float("nan")),
            })
    wt = A.wt_below_threshold(d)
    caption_extra = (f"{int((wt < A.PLDDT_ABS_THRESHOLD).sum())} of {len(wt)} "
                     f"wild types already predict below "
                     f"{A.PLDDT_ABS_THRESHOLD:.0f} pLDDT, so frac_below_abs80 "
                     f"is scaffold-dependent and is context, not a result. "
                     f"median_d_plddt is the median of per-scaffold medians.")
    write_table(t, "T7_plddt_dropoff",
                caption="Predicted-confidence drop-off by mutation load, "
                        "charge demand and decoding temperature. " + caption_extra)
    write_table(pd.DataFrame(cross), "T7b_plddt_first_crossing",
                caption="First bin on each axis where the median of "
                        "per-scaffold median ΔpLDDT against the scaffold's own "
                        "predicted wild type crosses each level.")
    return t

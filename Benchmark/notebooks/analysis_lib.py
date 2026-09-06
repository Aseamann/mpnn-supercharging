"""Shared base library for notebooks/analysis.ipynb.

PLAN.md Section 11. Loading with schema assertions, the house style, the
figure and table writers, and the statistics every figure balances with:
`plddt_frame`, `scaffold_balanced` and the pLDDT reference lines.

The figures themselves live in `figures_lib`, the tables in `tables_lib`, and
the definition of what the nine arms are and what they are called lives in
`figures_lib` so there is exactly one of it.

**The six-arm figure builders that used to live here were removed 2026-09-06**
(PLAN.md Section 0.1 item 25). They drew `figures/F1` to `F14` from the six-arm
run, and `results/` moved to nine arms on 2026-09-05. Rather than keep a second
figure set that could only get staler, the ten nine-arm figures in `figures_lib`
became the figure set. Git history has the removed code; `README.md` maps the
old numbering onto the new.

Reads only `results/*.csv` and `data/scaffold_manifest.csv`. Writes only
`figures/` and `tables/`. Nothing here touches the cluster, imports torch or
pyrosetta, or recomputes a metric that `10_aggregate.py` already wrote: if a
number is in a figure, it came from a CSV.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


TEXT = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#dedcd3"

# ---------------------------------------------------------------------------
# typography
#
# Myriad Pro, author's instruction 2026-09-06 (PLAN.md Section 0.1 item 22). It
# is a licensed Adobe font, so it is not in the conda env and is not committed
# to this repo. It is registered at runtime from wherever the files are, which
# keeps the notebook self-contained without redistributing the font.
#
# The search order lets another user point at their own copy through
# MPNN_BENCH_FONT_DIR, or drop the files into Benchmark/assets/fonts/, without
# editing this file. If nothing is found the figures fall back to DejaVu rather
# than failing, and set_style() prints what it resolved, so a figure set is
# never silently in the wrong font.
# ---------------------------------------------------------------------------

FONT_FAMILY = "Myriad Pro"
FONT_DIRS = [
    os.environ.get("MPNN_BENCH_FONT_DIR"),
    ROOT / "assets" / "fonts",
    Path.home() / "Fonts" / "MyriadPro",
]


def _register_fonts() -> list[str]:
    """Register any OTF/TTF found in FONT_DIRS, return the font.sans-serif stack."""
    found = set()
    for directory in FONT_DIRS:
        if not directory:
            continue
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.[ot]tf")):
            try:
                font_manager.fontManager.addfont(str(path))
                found.add(font_manager.get_font(str(path)).family_name)
            except Exception as exc:                              # noqa: BLE001
                warnings.warn(f"could not register {path}: {exc}")
    stack = [FONT_FAMILY] if FONT_FAMILY in found else []
    return stack + ["DejaVu Sans"]


def set_style() -> None:
    """House style. Called once from the notebook's first cell."""
    stack = _register_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": stack,
        # Type 42 embeds the outlines as TrueType. The default, Type 3, is
        # rejected by several journals and leaves the text uneditable in
        # Illustrator.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        # Publication sizes, raised 2026-09-06 on the author's instruction.
        # Axis labels and tick labels carry the reading of the figure at print
        # size, so they are the two that move most. Per-artist fontsize=
        # overrides in figures_lib beat these and were raised with them.
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.titlepad": 8.0, "axes.labelpad": 5.0,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.facecolor": "white",
    })
    if stack[0] != FONT_FAMILY:
        warnings.warn(
            f"{FONT_FAMILY} not found in {[str(d) for d in FONT_DIRS if d]}. "
            f"Figures will render in {stack[0]}.")
    print(f"style: font.sans-serif[0] = {stack[0]}, pdf.fonttype = 42")


def save(fig, name: str) -> None:
    """Write PNG and PDF at 300 dpi, per Section 11."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"wrote figures/{name}.png and .pdf")


def write_table(df: pd.DataFrame, name: str, caption: str = "",
                index: bool = False) -> None:
    """CSV plus LaTeX, per Section 11.2."""
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latex = df.to_latex(index=index, escape=True, longtable=False,
                            caption=caption or name, label=f"tab:{name}",
                            float_format="%.3f")
    (TABLES / f"{name}.tex").write_text(latex)
    print(f"wrote tables/{name}.csv and .tex  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# loading, with the schema assertions Section 11 asks for
# ---------------------------------------------------------------------------

REQUIRED = {
    "designs.csv": ["design_id", "scaffold_id", "fold_class", "method",
                    "delta_q_density", "target_charge", "q_actual", "hit_exact",
                    "n_mutations", "mut_per_charge", "d_reu_per_res",
                    "plddt_mean", "thread_tier"],
    "scaffold_summary.csv": ["scaffold_id", "method", "delta_q_density",
                             "hit_rate", "n_unique_mutation_sets",
                             "mean_pairwise_hamming", "positional_entropy",
                             "designable_coverage"],
    "rejection_curve.csv": ["scaffold_id", "delta_q_density", "n_on_target",
                            "hit_rate", "expected_samples_per_hit"],
    "rejection_distribution.csv": ["scaffold_id", "q_mean", "q_sd"],
}


def load(name: str, required: bool = True) -> pd.DataFrame | None:
    """Load one results CSV and assert its columns. Fails loudly, per Section 11."""
    path = RESULTS / name
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path} does not exist. Run the phase that writes it before "
                "this notebook; nothing here regenerates results.")
        print(f"note: {name} absent, sections depending on it will be skipped")
        return None
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED.get(name, []) if c not in df.columns]
    if missing:
        raise AssertionError(
            f"{name} is missing frozen columns {missing}. PLAN.md Section 10.1 is "
            "the authority: change it there, then the writer, then here.")
    return df


def load_all() -> dict:
    data = {
        "designs": load("designs.csv"),
        "summary": load("scaffold_summary.csv"),
        "curve": load("rejection_curve.csv"),
        "dist": load("rejection_distribution.csv"),
        "stats": load("statistics.csv", required=False),
        "runtime": load("runtime.csv", required=False),
        "manifest": pd.read_csv(ROOT / "data" / "scaffold_manifest.csv"),
        "esmfold": load("esmfold.csv", required=False),
        "af3": load("af3.csv", required=False),
    }
    d = data["designs"]
    print(f"designs.csv: {len(d)} designs, {d.scaffold_id.nunique()} scaffolds, "
          f"{d.method.nunique()} methods")
    print(f"  threaded: {d.thread_tier.notna().sum()}   "
          f"predicted: {d.plddt_mean.notna().sum()}")
    return data


def methods_present(df: pd.DataFrame) -> list[str]:
    """Methods in the frozen order, restricted to those actually in the data."""
    have = set(df["method"].dropna().unique())
    return [m for m in METHOD_ORDER if m in have]


FOCUS_CLASS_LABEL = "focus (eGFP)"


def with_fold_class(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the missing fold class of the focus scaffold.

    eGFP is added outside the CATH split and has no fold class, so every
    fold-class facet would silently drop the one scaffold the manuscript is
    actually about. It gets its own facet instead, labelled so nobody reads it
    as a CATH class.
    """
    out = df.copy()
    out["fold_class"] = out["fold_class"].fillna(FOCUS_CLASS_LABEL)
    return out


# ---------------------------------------------------------------------------
# F14 where predicted confidence starts to fall
#
# PLAN.md Section 11.1, added 2026-08-26. The question this answers was posed as
# "at what temperature or mutation count does pLDDT drop below 0.8". Asked of
# the absolute value it cannot be answered honestly: 10 of the 25 wild types
# already predict below 80, so an absolute cutoff mostly reports which scaffold
# was picked. Every panel here is therefore relative to each scaffold's own
# predicted wild type, and the absolute line is drawn only as context.
# ---------------------------------------------------------------------------

PLDDT_ABS_THRESHOLD = 80.0        # the 0.8 of the original question, 0 to 100
PLDDT_DROP_LEVELS = (-5.0, -10.0)
MUT_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 40, 60, 10 ** 6]
MUT_BIN_LABELS = ["1-5", "6-10", "11-15", "16-20", "21-25",
                  "26-30", "31-40", "41-60", ">60"]


def plddt_frame(data: dict) -> pd.DataFrame:
    """Designs carrying an ESMFold prediction, with the WT reference restored.

    `d_plddt_vs_wt_pred` is the design minus its scaffold's predicted wild type,
    so the wild-type value itself is recoverable by subtraction rather than
    needing a second file.
    """
    d = data["designs"]
    d = d.dropna(subset=["plddt_mean", "d_plddt_vs_wt_pred"]).copy()
    d["wt_pred_plddt"] = d["plddt_mean"] - d["d_plddt_vs_wt_pred"]
    d["abs_density"] = d["delta_q_density"].abs()
    d["mut_bin"] = pd.cut(d["n_mutations"], MUT_BIN_EDGES,
                          labels=MUT_BIN_LABELS, right=True)
    d["below_abs"] = d["plddt_mean"] < PLDDT_ABS_THRESHOLD
    return d


def wt_below_threshold(d: pd.DataFrame) -> pd.Series:
    """Predicted WT pLDDT per scaffold. One value each; medians guard against a
    scaffold whose rows disagree in the last decimal."""
    return d.groupby("scaffold_id")["wt_pred_plddt"].median().sort_values()


def _median_iqr(frame: pd.DataFrame, by: str, col: str):
    """Pooled median and IQR. Kept for the balanced axes only."""
    grp = frame.groupby(by, observed=True)[col]
    return grp.median(), grp.quantile(0.25), grp.quantile(0.75), grp.size()


MIN_SCAFFOLDS_PER_BIN = 5


def scaffold_balanced(frame: pd.DataFrame, by: str, col: str,
                      min_scaffolds: int = MIN_SCAFFOLDS_PER_BIN) -> pd.DataFrame:
    """Median of per-scaffold medians, with the spread taken across scaffolds.

    Pooling designs directly would let the axis carry the scaffold panel with
    it. The mutation-count axis is the clear case: only the longest scaffolds
    ever reach 40 or more mutations, and those are also the ones ESMFold
    predicts most confidently, so a pooled curve turns upward at the high end
    and reports which scaffolds reach that bin rather than what the mutations
    did. Taking each scaffold's median first gives every scaffold one vote.

    Bins holding fewer than `min_scaffolds` scaffolds are dropped rather than
    plotted thinly: they cannot support a median across scaffolds.
    """
    per = (frame.groupby(["scaffold_id", by], observed=True)[col]
                .median().reset_index())
    grp = per.groupby(by, observed=True)[col]
    out = pd.DataFrame({
        "median": grp.median(),
        "q25": grp.quantile(0.25),
        "q75": grp.quantile(0.75),
        "n_scaffolds": grp.size(),
    })
    n_designs = frame.groupby(by, observed=True)[col].size().rename("n_designs")
    out = out.join(n_designs)
    return out[out["n_scaffolds"] >= min_scaffolds]


def _reference_lines(ax):
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    for level in PLDDT_DROP_LEVELS:
        ax.axhline(level, color=GRID, lw=1.0, ls=":")



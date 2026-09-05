"""Figure and table builders for notebooks/analysis.ipynb.

PLAN.md Section 11. The notebook is the narrative; the drawing code lives here so
each figure is a named, testable function rather than a wall of cells, and so the
style is defined once.

Reads only `results/*.csv`. Writes only `figures/` and `tables/`. Nothing here
touches the cluster, imports torch or pyrosetta, or recomputes a metric that
`10_aggregate.py` already wrote: if a number is in a figure, it came from a CSV.

**Palette.** The six method colours are the first six slots of the `dataviz`
skill's documented categorical palette, in its documented order, which is the
order that carries its certification. `scripts/validate_palette.py` measures it
rather than assuming it, and reports:

    adjacent pairs   worst normal-vision dE 19.6 (floor 15), worst protan/deutan 9.1 (target 8)  PASS
    all pairs        worst normal-vision dE 12.9,             worst protan/deutan 3.2            FAIL

So six methods are safe in forms where only neighbouring series touch (lines,
grouped bars, boxplots) and are NOT safe in forms where any two series can land
side by side (scatter overlays). Figures of the second kind facet by method
instead of overlaying six colours. Colour is never the only channel: every
multi-series figure carries a legend, and markers differ by method as well.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

# Fixed order. A method keeps its colour whatever subset is plotted, so a figure
# that drops an arm never repaints the survivors.
METHOD_ORDER = ["mpnn_soluble", "mpnn_vanilla_weights",
                "mpnn_soluble_hbond_protected", "avnapsa", "rosetta",
                "random_control"]
METHOD_COLOR = dict(zip(METHOD_ORDER,
                        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                         "#e87ba4", "#008300"]))
METHOD_MARKER = dict(zip(METHOD_ORDER, ["o", "s", "^", "D", "v", "P"]))
METHOD_LABEL = {
    "mpnn_soluble": "MPNN supercharge (soluble)",
    "mpnn_vanilla_weights": "MPNN supercharge (original weights)",
    "mpnn_soluble_hbond_protected": "MPNN supercharge (h-bond protected)",
    "avnapsa": "AvNAPSA",
    "rosetta": "Rosetta supercharge",
    "random_control": "Random charge control",
}
VANILLA_COLOR = "#4a3aa7"     # slot 7, used only for the rejection sampler

# Der et al. 2013 reference values, annotated on F4 per Section 11.1.
DER_MUT_PER_CHARGE = {"AvNAPSA": 0.6, "Rosetta": 0.85}

TEXT = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#dedcd3"


def set_style() -> None:
    """House style. Called once from the notebook's first cell."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.facecolor": "white",
    })


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


def _style_for(method: str) -> dict:
    return {"color": METHOD_COLOR.get(method, MUTED),
            "marker": METHOD_MARKER.get(method, "o"),
            "label": METHOD_LABEL.get(method, method)}


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# F1  exact-hit rate vs |delta Q density|, faceted by fold class
# ---------------------------------------------------------------------------

def fig1(data: dict):
    d = with_fold_class(data["designs"])
    d = d.assign(abs_density=d["delta_q_density"].abs())
    classes = sorted(d["fold_class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(3.1 * len(classes), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fold in zip(axes, classes):
        sub = d[d["fold_class"] == fold]
        for method in methods_present(sub):
            m = sub[sub["method"] == method]
            grp = m.groupby("abs_density")["hit_exact"]
            xs, ys, los, his = [], [], [], []
            for density, series in grp:
                k, n = int(series.sum()), len(series)
                lo, hi = wilson(k, n)
                xs.append(density); ys.append(k / n); los.append(lo); his.append(hi)
            st = _style_for(method)
            ax.plot(xs, ys, **st)
            ax.fill_between(xs, los, his, color=st["color"], alpha=0.15, lw=0)
        ax.set_title(fold.replace("_", " "))
        ax.set_xlabel("|ΔQ density| (charge / 100 aa)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("exact-hit rate")
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("F1  Charge-targeting accuracy by fold class, Wilson 95% CI",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F2  the rejection-sampling ablation, the key figure for R3
# ---------------------------------------------------------------------------

def fig2(data: dict):
    """The R3 figure: what fraction of vanilla samples land on target.

    Zero-hit cells are the finding, and a log axis cannot show a zero. They are
    therefore counted in an annotation rather than drawn: plotting log(0) puts a
    spurious vertical drop to the axis floor at every affected scaffold, which
    reads as a measured decline instead of an absence. Per-scaffold traces are
    broken wherever a cell had no hit, for the same reason.
    """
    curve = data["curve"].copy()
    curve["abs_density"] = curve["delta_q_density"].abs()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for _, sub in curve.groupby("scaffold_id"):
        agg = sub.groupby("abs_density")["hit_rate"].mean()
        agg = agg.mask(agg <= 0)          # break the line, do not draw log(0)
        ax.plot(agg.index, agg.values, color=VANILLA_COLOR, alpha=0.22, lw=1.0,
                marker="", zorder=1)
    pooled = curve.groupby("abs_density").apply(
        lambda g: g["n_on_target"].sum() / g["n_samples_drawn"].sum())
    pooled_plot = pooled.mask(pooled <= 0)
    ax.plot(pooled_plot.index, pooled_plot.values, color=VANILLA_COLOR,
            marker="o", zorder=3, label="Vanilla ProteinMPNN, pooled over scaffolds")

    d = data["designs"]
    sc = d[d["method"] == "mpnn_soluble"].assign(
        abs_density=lambda x: x["delta_q_density"].abs())
    sc_rate = sc.groupby("abs_density")["hit_exact"].mean()
    ax.plot(sc_rate.index, sc_rate.values, color=METHOD_COLOR["mpnn_soluble"],
            marker="o", zorder=4, label="MPNN supercharge (soluble)")

    # How many cells produced nothing at all, stated as a number per rung.
    zero = (curve.assign(z=curve["n_on_target"] == 0)
            .groupby("abs_density")["z"].agg(["sum", "size"]))
    ymin = min(pooled_plot.min(), 1.0 / curve["n_samples_drawn"].max()) * 0.35
    ax.set_ylim(ymin, 1.9)
    for density, row in zero.iterrows():
        if row["sum"] == 0:
            continue
        ax.annotate(f"{int(row['sum'])}/{int(row['size'])}\ncells\n0 hits",
                    xy=(density, ymin * 1.25), ha="center", va="bottom",
                    fontsize=7, color=VANILLA_COLOR, linespacing=1.25)

    n_samples = int(curve["n_samples_drawn"].max())
    ax.set_yscale("log")
    ax.set_xlabel("|ΔQ density| (charge / 100 aa)")
    ax.set_ylabel(f"fraction of {n_samples:,} samples on target (log)")
    ax.set_title("F2  Rejection sampling cannot reach the targets the method hits")
    # Legend below the axes: the zero-hit annotations occupy the bottom of the
    # plotting area and any in-axes placement collided with them.
    from matplotlib.lines import Line2D                            # noqa: PLC0415
    handles = [
        Line2D([], [], color=METHOD_COLOR["mpnn_soluble"], marker="o",
               label="MPNN supercharge (soluble)"),
        Line2D([], [], color=VANILLA_COLOR, marker="o",
               label="Vanilla ProteinMPNN, pooled over scaffolds"),
        Line2D([], [], color=VANILLA_COLOR, alpha=0.35, lw=1.0,
               label="individual scaffolds (broken where a cell had no hit)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=1, handlelength=2.4)
    fig.tight_layout()
    return fig


def fig2_inset_cost(data: dict):
    """The F2 inset: expected CPU-seconds per on-target design."""
    curve = data["curve"].copy()
    curve["abs_density"] = curve["delta_q_density"].abs()
    known = curve[~curve["expected_gpu_seconds_per_hit"].astype(str).str.startswith(">")]
    known = known.assign(cost=known["expected_gpu_seconds_per_hit"].astype(float))
    mpnn = curve[curve["mpnn_sc_gpu_seconds_per_hit"].astype(str).str.startswith(">") == False]
    mpnn = mpnn.assign(cost=mpnn["mpnn_sc_gpu_seconds_per_hit"].astype(float))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    a = known.groupby("abs_density")["cost"].median()
    b = mpnn.groupby("abs_density")["cost"].median()
    ax.plot(a.index, a.values, color=VANILLA_COLOR, marker="o",
            label="Vanilla rejection (cells with ≥1 hit)")
    ax.plot(b.index, b.values, color=METHOD_COLOR["mpnn_soluble"], marker="o",
            label="MPNN supercharge")
    ax.set_yscale("log")
    ax.set_xlabel("|ΔQ density|")
    ax.set_ylabel("CPU-seconds per on-target design")
    ax.set_title("F2 inset  Cost per accepted design", fontsize=9)
    n_cens = int((curve["expected_gpu_seconds_per_hit"].astype(str)
                  .str.startswith(">")).sum())
    ax.text(0.02, 0.02, f"{n_cens} of {len(curve)} cells censored:\n"
                        "rejection produced no hit, so no ratio exists",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F3  empirical charge distribution of the vanilla pools
# ---------------------------------------------------------------------------

def fig3(data: dict):
    dist = data["dist"].sort_values("q_mean")
    curve = data["curve"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ypos = np.arange(len(dist))
    ax.errorbar(dist["q_mean"], ypos,
                xerr=1.96 * dist["q_sd"], fmt="o", color=VANILLA_COLOR,
                ecolor=VANILLA_COLOR, elinewidth=2, capsize=0, markersize=4,
                label="vanilla sample mean ± 1.96 sd")
    for y, scaffold in zip(ypos, dist["scaffold_id"]):
        targets = curve[curve["scaffold_id"] == scaffold]
        hit = targets[targets["n_on_target"] > 0]["target_charge"]
        miss = targets[targets["n_on_target"] == 0]["target_charge"]
        ax.scatter(miss, np.full(len(miss), y), marker="x", s=22,
                   color="#e34948", zorder=3)
        ax.scatter(hit, np.full(len(hit), y), marker="|", s=44,
                   color="#008300", zorder=3)
    ax.scatter([], [], marker="x", color="#e34948", label="target, 0 of 2000 sampled")
    ax.scatter([], [], marker="|", color="#008300", label="target, ≥1 sampled")
    ax.scatter([], [], marker="o", color=VANILLA_COLOR, label="WT net charge")
    ax.scatter(dist["q_wt"], ypos, marker="o", s=14, facecolor="white",
               edgecolor=TEXT, zorder=4, linewidths=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(dist["scaffold_id"], fontsize=6.5)
    ax.set_xlabel("net charge")
    ax.set_title("F3  Where the charge ladder sits relative to what vanilla "
                 "ProteinMPNN actually samples")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F4  mutations per unit charge
# ---------------------------------------------------------------------------

def fig4(data: dict):
    d = with_fold_class(data["designs"]).dropna(subset=["mut_per_charge"])
    classes = sorted(d["fold_class"].unique())
    order = methods_present(d)
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fold in zip(axes, classes):
        sub = d[d["fold_class"] == fold]
        boxes = [sub[sub["method"] == m]["mut_per_charge"].values for m in order]
        bp = ax.boxplot(boxes, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": TEXT, "linewidth": 1.4})
        for patch, method in zip(bp["boxes"], order):
            patch.set_facecolor(METHOD_COLOR[method])
            patch.set_alpha(0.75)
            patch.set_edgecolor(METHOD_COLOR[method])
        for key in ("whiskers", "caps"):
            for artist in bp[key]:
                artist.set_color(MUTED)
        for value, name, ls in ((DER_MUT_PER_CHARGE["AvNAPSA"], "Der AvNAPSA", ":"),
                                (DER_MUT_PER_CHARGE["Rosetta"], "Der Rosetta", "--")):
            ax.axhline(value, color=MUTED, ls=ls, lw=1.0)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([METHOD_LABEL[m].replace(" (", "\n(") for m in order],
                           rotation=45, ha="right", fontsize=6.5)
        ax.set_title(fold.replace("_", " "))
    axes[0].set_ylabel("mutations per unit charge moved")
    axes[-1].text(0.98, 0.97, "dotted: Der AvNAPSA 0.6\ndashed: Der Rosetta 0.85",
                  transform=axes[-1].transAxes, fontsize=7, color=MUTED,
                  ha="right", va="top")
    fig.suptitle("F4  Mutational efficiency", y=1.03, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F5  diversity panel
# ---------------------------------------------------------------------------

def fig5(data: dict):
    s = data["summary"]
    panels = [("n_unique_mutation_sets", "unique mutation sets (of n)"),
              ("mean_pairwise_hamming", "mean pairwise Hamming\n(designable positions)"),
              ("positional_entropy", "positional entropy (bits)"),
              ("designable_coverage", "designable coverage")]
    order = methods_present(s)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))
    for ax, (col, label) in zip(axes, panels):
        boxes = [s[s["method"] == m][col].dropna().values for m in order]
        bp = ax.boxplot(boxes, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": TEXT, "linewidth": 1.4})
        for patch, method in zip(bp["boxes"], order):
            patch.set_facecolor(METHOD_COLOR[method])
            patch.set_alpha(0.75)
            patch.set_edgecolor(METHOD_COLOR[method])
        for key in ("whiskers", "caps"):
            for artist in bp[key]:
                artist.set_color(MUTED)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([m.replace("mpnn_", "").replace("_", "\n")
                            for m in order], rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel(label, fontsize=8)
    fig.suptitle("F5  Sequence diversity per scaffold × target cell "
                 "(AvNAPSA has one design per cell, so its first two panels are "
                 "undefined by construction)", y=1.06, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F6  delta REU per residue
# ---------------------------------------------------------------------------

def fig6(data: dict):
    d = data["designs"].dropna(subset=["d_reu_per_res"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for method in methods_present(d):
        m = d[d["method"] == method]
        grp = m.groupby("delta_q_density")["d_reu_per_res"]
        med = grp.median()
        lo = grp.quantile(0.25)
        hi = grp.quantile(0.75)
        st = _style_for(method)
        ax.plot(med.index, med.values, **st)
        ax.fill_between(med.index, lo.values, hi.values, color=st["color"],
                        alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("ΔQ density (charge / 100 aa)")
    ax.set_ylabel("Δ REU per residue vs relaxed WT")
    ax.set_title("F6  Energetic cost, median with interquartile ribbon")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F7  per-term decomposition, styled after Der et al. Figure 5
# ---------------------------------------------------------------------------

def fig7(data: dict):
    d = data["designs"]
    terms = ["d_fa_atr", "d_fa_rep", "d_fa_sol", "d_fa_elec",
             "d_hbond_sc", "d_hbond_bb_sc"]
    have = [t for t in terms if t in d.columns and d[t].notna().any()]
    if not have:
        return None
    order = methods_present(d.dropna(subset=have[:1]))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    width = 0.8 / max(len(order), 1)
    x = np.arange(len(have))
    for i, method in enumerate(order):
        m = d[d["method"] == method]
        vals = [m[t].median() for t in have]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
               color=METHOD_COLOR[method], label=METHOD_LABEL[method],
               edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("d_", "") for t in have])
    ax.set_ylabel("median Δ term per residue")
    ax.set_title("F7  Per-term energy decomposition (analogue of Der et al. Fig. 5)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F8  recovery and REU spread vs target charge  (Reviewer 3's requested plot)
# ---------------------------------------------------------------------------

def fig8(data: dict, method: str = "mpnn_soluble"):
    """Twin-axis, as Section 11.1 specifies.

    A twin y-axis is normally the wrong choice: two scales invite a false
    reading of where the curves cross. It is used here because Section 11.1
    names this plot as the one Reviewer 3 asked for, and changing the form would
    answer a different request. The two axes are colour-keyed to their curves and
    the crossing point carries no meaning; `fig8_small_multiples` draws the same
    data on separate panels for anyone who prefers it.
    """
    d = data["designs"]
    m = d[d["method"] == method]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    grp = m.groupby("delta_q_density")
    rec = grp["seq_identity_to_wt_designable"].median()
    ax.plot(rec.index, rec.values, color=METHOD_COLOR[method], marker="o",
            label="sequence identity at designable positions")
    ax.set_xlabel("ΔQ density (charge / 100 aa)")
    ax.set_ylabel("identity to WT at designable positions",
                  color=METHOD_COLOR[method])
    ax.tick_params(axis="y", colors=METHOD_COLOR[method])

    if m["d_reu_per_res"].notna().any():
        ax2 = ax.twinx()
        spread = grp["d_reu_per_res"].agg(lambda s: s.quantile(0.75) - s.quantile(0.25))
        ax2.plot(spread.index, spread.values, color="#eb6834", marker="s", ls="--",
                 label="ΔREU per residue, IQR spread")
        ax2.set_ylabel("ΔREU per residue, IQR", color="#eb6834")
        ax2.tick_params(axis="y", colors="#eb6834")
        ax2.grid(False)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], frameon=False,
                  fontsize=7.5, loc="lower left")
    ax.set_title(f"F8  Recovery and energetic spread vs target charge "
                 f"({METHOD_LABEL[method]})")
    fig.tight_layout()
    return fig


def fig8_small_multiples(data: dict, method: str = "mpnn_soluble"):
    """The same data as F8 on two panels, with no shared axis to misread."""
    d = data["designs"]
    m = d[d["method"] == method]
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.0), sharex=True)
    grp = m.groupby("delta_q_density")
    rec = grp["seq_identity_to_wt_designable"].median()
    axes[0].plot(rec.index, rec.values, color=METHOD_COLOR[method], marker="o")
    axes[0].set_ylabel("identity to WT\n(designable)")
    spread = grp["d_reu_per_res"].agg(lambda s: s.quantile(0.75) - s.quantile(0.25))
    axes[1].plot(spread.index, spread.values, color="#eb6834", marker="s")
    axes[1].set_ylabel("ΔREU per residue\nIQR")
    axes[1].set_xlabel("ΔQ density (charge / 100 aa)")
    fig.suptitle(f"F8 alt  Same data, separate panels ({METHOD_LABEL[method]})",
                 fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F9  hydrogen bonds
# ---------------------------------------------------------------------------

def fig9(data: dict):
    d = data["designs"]
    if "d_hbond_strong" not in d.columns or d["d_hbond_strong"].isna().all():
        return None
    d = d.dropna(subset=["d_hbond_strong"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharex=True)
    for ax, col, title in ((axes[0], "d_hbond_strong", "strong (≤ −0.5 REU)"),
                           (axes[1], "d_hbond_weak", "weak (−0.5 to −0.1 REU)")):
        if col not in d.columns:
            continue
        for method in methods_present(d):
            m = d[d["method"] == method]
            med = m.groupby("delta_q_density")[col].median()
            st = _style_for(method)
            ax.plot(med.index, med.values, **st)
        ax.axhline(0, color=MUTED, lw=1.0, ls="--")
        ax.set_title(f"F9  Δ side-chain H-bonds, {title}")
        ax.set_xlabel("ΔQ density")
    axes[0].set_ylabel("Δ hydrogen bonds vs relaxed WT")
    axes[1].legend(frameon=False, fontsize=7.5,
                   loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F10 structure prediction.  Faceted, because 6 overlaid series fail the
#     all-pairs colour check.  See this module's docstring.
# ---------------------------------------------------------------------------

def fig10(data: dict):
    d = data["designs"]
    if d["plddt_mean"].isna().all():
        return None
    d = d.dropna(subset=["plddt_mean"])
    order = methods_present(d)
    af3 = data.get("af3")
    fig, axes = plt.subplots(2, len(order), figsize=(2.3 * len(order), 5.2),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for j, method in enumerate(order):
        m = d[d["method"] == method]
        colour = METHOD_COLOR[method]
        for i, col in enumerate(("d_plddt_vs_wt_pred", "ca_rmsd_to_wt_crystal")):
            ax = axes[i, j]
            if col not in m.columns or m[col].isna().all():
                continue
            grp = m.groupby("delta_q_density")[col]
            med, lo, hi = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
            ax.plot(med.index, med.values, color=colour, marker=METHOD_MARKER[method])
            ax.fill_between(med.index, lo.values, hi.values, color=colour,
                            alpha=0.15, lw=0)
            if i == 0:
                ax.axhline(0, color=MUTED, lw=1.0, ls="--")
                ax.set_title(METHOD_LABEL[method].replace(" (", "\n("), fontsize=7.5)
        if af3 is not None and not af3.empty:
            a = af3[(af3["method"] == method) & (af3["status"] == "ok")]
            if len(a) and "af3_ca_rmsd" in a.columns:
                axes[1, j].scatter(a["delta_q_density"], a["af3_ca_rmsd"],
                                   s=44, facecolor="white", edgecolor=TEXT,
                                   zorder=5, linewidths=1.0, label="AF3")
    axes[0, 0].set_ylabel("Δ pLDDT vs WT prediction")
    axes[1, 0].set_ylabel("CA RMSD to WT crystal (Å)")
    for ax in axes[1]:
        ax.set_xlabel("ΔQ density")
    fig.suptitle("F10  Independent structure prediction. ESMFold lines with IQR "
                 "ribbon; AF3 subset as open markers", y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F11 charge-capacity ceiling
# ---------------------------------------------------------------------------

def fig11(data: dict):
    d = data["designs"]
    prim = d[d["method"] == "mpnn_soluble"].copy()
    manifest = data["manifest"].set_index("scaffold_id")

    # final_temperature is empty for every non-MPNN arm, so the column arrives as
    # mixed str/float and any ordered operation on it raises. Coerce once here
    # rather than at each use; the baselines are already filtered out above, but
    # the column dtype is set by the whole frame, not by this slice.
    prim["final_temperature"] = pd.to_numeric(prim["final_temperature"],
                                              errors="coerce")

    # Decision E: the ceiling is the largest |density| hit at final_temperature 0.3.
    at_base = prim[(prim["hit_exact"]) & (prim["final_temperature"] == 0.3)]
    ceiling = (at_base.assign(a=at_base["delta_q_density"].abs())
               .groupby("scaffold_id")["a"].max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    # eGFP is the focus scaffold, added outside the CATH split, so it has no
    # fold class and the column mixes NaN with strings. It is labelled rather
    # than dropped: it is the scaffold the manuscript's Figure 2A is about, and
    # this figure exists to generalise that figure, so leaving it out would be
    # the one omission a reader would notice.
    classes = manifest.loc[ceiling.index, "fold_class"].fillna(FOCUS_CLASS_LABEL)
    class_names = sorted(classes.unique())
    for i, fold in enumerate(class_names):
        vals = ceiling[classes == fold]
        axes[0].scatter(np.full(len(vals), i) + np.random.default_rng(0)
                        .uniform(-0.12, 0.12, len(vals)),
                        vals, s=26, color=METHOD_COLOR["mpnn_soluble"], alpha=0.8)
        axes[0].hlines(vals.median(), i - 0.25, i + 0.25, color=TEXT, lw=2)
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels([c.replace("_", " ") for c in class_names])
    axes[0].set_ylabel("max |ΔQ density| reached at T = 0.3")
    axes[0].set_title("Per-scaffold charge capacity")

    pivot = prim.pivot_table(index="scaffold_id", columns="delta_q_density",
                             values="final_temperature", aggfunc="max")
    im = axes[1].imshow(pivot.values, aspect="auto", cmap="YlGnBu",
                        vmin=0.3, vmax=0.9)
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels(pivot.columns)
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index, fontsize=6)
    axes[1].set_xlabel("ΔQ density")
    axes[1].set_title("Final temperature required")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], label="final temperature")
    fig.suptitle("F11  Charge capacity ceiling, manuscript Fig. 2A generalised "
                 f"from one protein to {len(ceiling)}", y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F12 runtime
# ---------------------------------------------------------------------------

def fig12(data: dict):
    d = data["designs"]
    order = methods_present(d)
    per_cell = (d.groupby(["method", "scaffold_id", "target_charge"])
                .agg(wall=("wall_seconds", "first"),
                     hits=("hit_exact", "sum")).reset_index())
    per_cell = per_cell[per_cell["hits"] > 0]
    per_cell["cost"] = per_cell["wall"] / per_cell["hits"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    boxes = [per_cell[per_cell["method"] == m]["cost"].dropna().values
             for m in order]
    bp = ax.boxplot(boxes, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={"color": TEXT, "linewidth": 1.4})
    for patch, method in zip(bp["boxes"], order):
        patch.set_facecolor(METHOD_COLOR[method])
        patch.set_alpha(0.75)
        patch.set_edgecolor(METHOD_COLOR[method])
    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color(MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([METHOD_LABEL[m].replace(" (", "\n(") for m in order],
                       rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("CPU-seconds per on-target design (log)")
    ax.set_title("F12  Runtime per accepted design, retries included\n"
                 "Cells with no on-target design are excluded: they have no "
                 "cost per hit to plot", fontsize=9)
    fig.tight_layout()
    return fig


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


def _plot_balanced(ax, frame, by, method, x_of=None):
    bal = scaffold_balanced(frame, by, "d_plddt_vs_wt_pred")
    if bal.empty:
        return False
    x = [x_of(v) for v in bal.index] if x_of else list(bal.index)
    ax.plot(x, bal["median"].values, color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method], ms=4, lw=1.4,
            label=METHOD_LABEL[method])
    ax.fill_between(x, bal["q25"].values, bal["q75"].values,
                    color=METHOD_COLOR[method], alpha=0.12, lw=0)
    return True


def _reference_lines(ax):
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    for level in PLDDT_DROP_LEVELS:
        ax.axhline(level, color=GRID, lw=1.0, ls=":")


def fig14(data: dict):
    d = plddt_frame(data)
    if d.empty:
        return None
    order = methods_present(d)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))

    # (a) against mutation count
    ax = axes[0, 0]
    for method in order:
        _plot_balanced(ax, d[d["method"] == method], "mut_bin", method,
                       x_of=lambda v: MUT_BIN_LABELS.index(str(v)))
    _reference_lines(ax)
    ax.set_xticks(range(len(MUT_BIN_LABELS)))
    ax.set_xticklabels(MUT_BIN_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("mutations")
    ax.set_ylabel("Δ pLDDT vs WT prediction")
    ax.set_title("a  against mutation count", fontsize=9, loc="left")

    # (b) against charge demand
    ax = axes[0, 1]
    for method in order:
        _plot_balanced(ax, d[d["method"] == method], "abs_density", method)
    _reference_lines(ax)
    ax.set_xlabel("|ΔQ density|")
    ax.set_ylabel("Δ pLDDT vs WT prediction")
    ax.set_title("b  against charge demand", fontsize=9, loc="left")

    # (c) against decoding temperature, MPNN arms only: it is the only place
    # final_temperature is defined.
    ax = axes[1, 0]
    mpnn = d[d["method"].str.startswith("mpnn")].dropna(subset=["final_temperature"])
    drawn = False
    for method in [m for m in order if m.startswith("mpnn")]:
        drawn |= _plot_balanced(ax, mpnn[mpnn["method"] == method],
                                "final_temperature", method)
    if drawn:
        _reference_lines(ax)
    else:
        ax.text(0.5, 0.5, "no temperature bin holds "
                          f"{MIN_SCAFFOLDS_PER_BIN} scaffolds", ha="center",
                va="center", transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.set_xlabel("final decoding temperature")
    ax.set_ylabel("Δ pLDDT vs WT prediction")
    ax.set_title("c  against decoding temperature (MPNN arms)", fontsize=9,
                 loc="left")

    # (d) the confound: where the wild types already sit
    ax = axes[1, 1]
    wt = wt_below_threshold(d)
    colours = [METHOD_COLOR["avnapsa"] if v < PLDDT_ABS_THRESHOLD else MUTED
               for v in wt.values]
    ax.bar(range(len(wt)), wt.values, color=colours, width=0.8)
    ax.axhline(PLDDT_ABS_THRESHOLD, color=TEXT, lw=1.2, ls="--")
    n_below = int((wt < PLDDT_ABS_THRESHOLD).sum())
    ax.annotate(f"{n_below} of {len(wt)} wild types already below "
                f"{PLDDT_ABS_THRESHOLD:.0f}",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7,
                color=TEXT)
    ax.set_xticks(range(len(wt)))
    ax.set_xticklabels(wt.index, rotation=90, fontsize=5.5)
    ax.set_ylabel("predicted WT pLDDT")
    ax.set_title("d  why the absolute cutoff is scaffold-dependent",
                 fontsize=9, loc="left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("F14  Where predicted confidence starts to fall. ESMFold2, "
                 "median of per-scaffold medians with the IQR taken across "
                 f"scaffolds, bins holding at least {MIN_SCAFFOLDS_PER_BIN} "
                 "scaffolds", y=1.0, fontsize=10)
    fig.tight_layout()
    return fig

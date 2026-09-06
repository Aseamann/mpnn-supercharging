"""Figure builders for notebooks/analysis.ipynb.

PLAN.md Section 11.4. These ten figures cover the nine arms and are the figure
set; the six-arm figures that `analysis_lib.py` used to build were retired on
2026-09-06 (PLAN.md Section 0.1 item 25) once `results/` had moved to the nine
arms and keeping a second, stale set had no reader.

Loading, styling and the shared statistics live in `analysis_lib`, which is now
the base library rather than a parallel figure set. `tables_lib` takes its arm
order and labels from here, so there is one definition of what the nine arms
are and what they are called.

The Section 11 constraint applies: reads only `results/*.csv` and
`data/scaffold_manifest.csv`, writes only `figures/` and `tables/`, touches no
cluster, and recomputes no metric that `10_aggregate.py` already wrote.

**Titles carry no context.** Every title is the short form `F2  Mutational
efficiency` and the context lives in a markdown Notes cell under each figure in
the notebook, at greater length than a title could hold.

**Palette.** Nine arms cannot be nine distinct colourblind-safe hues. The
palette groups arms into one hue family per method family, with lightness
separating members inside a family, so the distinctions the eye has to make
between neighbours are the ones the adjacent-pair check measures.
`scripts/validate_palette.py` measures it rather than assuming it, and reports:

    adjacent pairs   worst normal-vision dE 15.9 (floor 15), worst protan/deutan 14.7 (target 8)  PASS
    all pairs        worst normal-vision dE 11.9 (FAIL),     worst protan/deutan  6.8 (WARN, floor 6)

Reproduce with:

    python scripts/validate_palette.py --colors "$(python -c "
    import figures_lib as U; print(','.join(U.PALETTE))")" --pairs adjacent

So neighbouring series are safe in lines, grouped bars and boxplots; any two
series landing side by side are not, which is why F7 facets by method instead
of overlaying. Colour is never the only channel: every multi-series figure
carries a legend and markers differ by method.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_lib as A
from analysis_lib import (  # noqa: F401  (re-exported for the notebook)
    GRID, MUTED, TEXT, load_all, set_style, with_fold_class, write_table,
)

FIGURE_DIR = A.ROOT / "figures"

# ---------------------------------------------------------------------------
# arms and palette
# ---------------------------------------------------------------------------

# Family order, and inside each family dark to light. Adjacent pairs are what
# the palette check certifies, so the order here is part of the certification.
METHOD_FAMILY = {
    "mpnn_soluble": "MPNN supercharge, stock weights",
    "mpnn_vanilla_weights": "MPNN supercharge, stock weights",
    "mpnn_soluble_hbond_protected": "MPNN supercharge, stock weights",
    "mpnn_hyper": "MPNN supercharge, biased weights",
    "mpnn_halo": "MPNN supercharge, biased weights",
    "avnapsa": "classical",
    "rosetta": "classical",
    "rosetta_hbond_off": "classical",
    "random_control": "control",
    "vanilla_rejection": "control",
}

METHOD_ORDER = ["mpnn_soluble", "mpnn_vanilla_weights",
                "mpnn_soluble_hbond_protected", "mpnn_hyper", "mpnn_halo",
                "avnapsa", "rosetta", "rosetta_hbond_off", "random_control"]

# vanilla_rejection is not an arm of designs.csv. It is a pool, and it appears
# only where a pool can answer the question: F1, F2 and F8.
VANILLA = "vanilla_rejection"
ALL_SERIES = METHOD_ORDER + [VANILLA]

PALETTE = ["#12406f", "#2a78d6", "#86b6ec",      # stock-weight MPNN, blues
           "#0b6e63", "#35b3a1",                  # biased-weight MPNN, teals
           "#9a4a00", "#e08214", "#f6c26b",       # classical, ambers
           "#8a8880",                             # random control, grey
           "#4a3aa7"]                             # unguided pool, violet

METHOD_COLOR = dict(zip(ALL_SERIES, PALETTE))
METHOD_MARKER = dict(zip(ALL_SERIES,
                         ["o", "s", "^", "D", "v", "P", "X", "*", "h", "d"]))
METHOD_LABEL = {
    "mpnn_soluble": "MPNN supercharge (soluble)",
    "mpnn_vanilla_weights": "MPNN supercharge (original weights)",
    "mpnn_soluble_hbond_protected": "MPNN supercharge (h-bond protected)",
    "mpnn_hyper": "MPNN supercharge (HyperMPNN)",
    "mpnn_halo": "MPNN supercharge (HaloMPNN)",
    "avnapsa": "AvNAPSA",
    "rosetta": "Rosetta supercharge",
    "rosetta_hbond_off": "Rosetta supercharge (h-bond off)",
    "random_control": "Random charge control",
    VANILLA: "Unguided MPNN (rejection pool)",
}

# Short forms for boxplot tick labels. The full labels are two lines of up to
# 35 characters and ten of them across a facet overlap into illegibility; the
# legends and notes carry the long form.
METHOD_SHORT = {
    "mpnn_soluble": "MPNN soluble",
    "mpnn_vanilla_weights": "MPNN original",
    "mpnn_soluble_hbond_protected": "MPNN hb-protected",
    "mpnn_hyper": "HyperMPNN",
    "mpnn_halo": "HaloMPNN",
    "avnapsa": "AvNAPSA",
    "rosetta": "Rosetta",
    "rosetta_hbond_off": "Rosetta hb-off",
    "random_control": "random control",
    VANILLA: "unguided pool",
}

# The five arms where final_temperature exists. It is written by the `-u`
# escalation loop in protein_mpnn_supercharge.py and is blank everywhere else.
MPNN_ARMS = [m for m in METHOD_ORDER if m.startswith("mpnn")]

# Der et al. 2013 reference values, annotated on F2.
DER_MUT_PER_CHARGE = {"AvNAPSA": 0.6, "Rosetta": 0.85}

# F2 facets, fixed. analysis_lib.fig2 used sorted(), which orders them
# alphabetically as alpha_beta, focus (eGFP), mainly_alpha, mainly_beta. eGFP is
# one scaffold and does not deserve a facet of its own next to eight-scaffold
# classes; it is carried by the pooled facet instead.
FOLD_FACETS = ["mainly_alpha", "mainly_beta", "alpha_beta"]
POOLED_FACET = "all 25 scaffolds"


def save(fig, name: str) -> None:
    """PNG and PDF at 300 dpi into figures/."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}")
    print(f"wrote figures/{name}.png and .pdf")


def methods_present(df: pd.DataFrame) -> list[str]:
    """Arms in the fixed family order, restricted to those in the data."""
    have = set(df["method"].dropna().unique())
    return [m for m in METHOD_ORDER if m in have]


def _style_for(method: str) -> dict:
    return {"color": METHOD_COLOR.get(method, MUTED),
            "marker": METHOD_MARKER.get(method, "o"),
            "label": METHOD_LABEL.get(method, method)}


def _tick_labels(order: list[str]) -> list[str]:
    return [METHOD_SHORT.get(m, m) for m in order]


def _style_boxes(bp, order: list[str]) -> None:
    for patch, method in zip(bp["boxes"], order):
        patch.set_facecolor(METHOD_COLOR[method])
        patch.set_alpha(0.75)
        patch.set_edgecolor(METHOD_COLOR[method])
    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color(MUTED)


def _legend_right(ax, **kw):
    """The house convention for a legend that must leave the data alone.

    Silent no-op on an axis with nothing labelled, which happens when a panel's
    arms are all absent from the data. Matplotlib warns there, and the warning
    would be the only sign; the panel itself already says what is missing.
    """
    handles, _ = ax.get_legend_handles_labels()
    if not handles and "handles" not in kw:
        return
    ax.legend(frameon=False, fontsize=10, loc="center left",
              bbox_to_anchor=(1.02, 0.5), **kw)


def load_updated() -> dict:
    """`analysis_lib.load_all` plus the unguided-MPNN hit table and T sweep."""
    data = load_all()
    data["hits"] = A.load("rejection_hits.csv", required=False)
    if data["hits"] is not None:
        h = data["hits"]
        print(f"rejection_hits.csv: {len(h)} on-target unguided samples, "
              f"{h.scaffold_id.nunique()} scaffolds, "
              f"rungs {sorted(h.delta_q_density.unique())}")
    data["curve_by_t"] = load_rejection_sweep(data)
    return data


# The temperature the existing pools were drawn at. config/benchmark.yaml is
# the source of truth for the sampler, but the notebook is not allowed to read
# it, so the value is asserted against the CSV instead: any curve file without a
# sampling_temperature column is the default pool, and every file that has one
# states its own temperature.
BASE_TEMPERATURE = 0.3


def load_rejection_sweep(data: dict) -> pd.DataFrame:
    """Every rejection curve on disk, stacked, with a sampling_temperature column.

    PLAN.md Section 0.1 item 23. `rejection_curve.csv` is the T = 0.3 pool and
    carries no temperature column, so it is labelled here; each
    `rejection_curve_T*.csv` carries its own. Missing sweep files are not an
    error: F1 then draws the temperatures that exist and says so, rather than
    the notebook failing or, worse, a tier being drawn from nothing.
    """
    frames = []
    base = data["curve"]
    if base is not None:
        frames.append(base.assign(sampling_temperature=BASE_TEMPERATURE))
    for path in sorted(A.RESULTS.glob("rejection_curve_T*.csv")):
        frame = pd.read_csv(path)
        if "sampling_temperature" not in frame.columns:
            raise ValueError(f"{path.name} has no sampling_temperature column")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    counts = out.groupby("sampling_temperature").size().to_dict()
    print("rejection curves by sampling temperature: "
          + ", ".join(f"T={t:g}: {n} cells" for t, n in sorted(counts.items())))
    return out


# ---------------------------------------------------------------------------
# F1  where the ladder sits relative to what unguided ProteinMPNN samples
#
# Reworked: the legend is off the data, the +-1.96 sd error bars become a light
# band so 25 scaffold rows stay readable, and the sample count in the legend is
# read from the data instead of hard-coded.
#
# AMENDED 2026-09-06 on the author's instruction, PLAN.md Section 0.1 item 23.
# The pools now exist at three temperatures, so each rung's marker states the
# LOWEST temperature at which unguided sampling reached it rather than a plain
# hit or miss. The previous two-tier version reported the reach of T = 0.3 while
# reading as the reach of unguided sampling in general, which is the question
# reviewer R3 actually asks.
#
# What a red x means is bounded and the notes cell says so: not reached in 2,000
# draws at any of T = 0.3, 0.6, 0.9. It is not a claim that the rung is
# unreachable, and a deeper pool could move a marker.
# ---------------------------------------------------------------------------

# Author's specification, 2026-09-06. Ordered coldest first; the tier assigned
# to a cell is the first entry whose pool reached it.
REACH_TIERS = [
    (0.3, "#006c31", "reached at T = 0.3"),
    (0.6, "#e8c33d", "first reached at T = 0.6"),
    (0.9, "#c1272d", "first reached at T = 0.9"),
]
REACH_MISS_COLOR = "#c1272d"


def reach_tiers(curve_by_t: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Per (scaffold, target), the lowest temperature whose pool hit it.

    `reached_at` is NaN where no pool at any available temperature landed on the
    rung. Only temperatures actually on disk are considered, and the list of
    them is returned so the legend describes the pools that exist rather than
    the ones the figure was designed against.
    """
    have = sorted(curve_by_t["sampling_temperature"].unique())
    hit = curve_by_t[curve_by_t["n_on_target"] > 0]
    lowest = (hit.groupby(["scaffold_id", "target_charge"])["sampling_temperature"]
              .min().rename("reached_at"))
    cells = (curve_by_t[["scaffold_id", "target_charge"]]
             .drop_duplicates()
             .merge(lowest, on=["scaffold_id", "target_charge"], how="left"))
    return cells, have


def fig1(data: dict):
    dist = data["dist"].sort_values("q_mean").reset_index(drop=True)
    curve = data["curve"]
    curve_by_t = data.get("curve_by_t")
    if curve_by_t is None:
        curve_by_t = load_rejection_sweep(data)
    cells, have = reach_tiers(curve_by_t)
    tiers = [t for t in REACH_TIERS if any(abs(t[0] - h) < 1e-9 for h in have)]

    drawn = sorted(curve["n_samples_drawn"].dropna().unique())
    # analysis_lib.fig1 wrote "0 of 2000" as a literal. Read it instead: if the
    # pools ever run at a different depth the legend must say so, not lie.
    n_label = (f"{int(drawn[0])}" if len(drawn) == 1
               else f"{int(min(drawn))} to {int(max(drawn))}")
    t_label = ", ".join(f"{t:g}" for t in have)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ypos = np.arange(len(dist))
    colour = METHOD_COLOR[VANILLA]

    # The sampled mass, as a band rather than error bars with caps. 25 rows of
    # capped bars plus two marker types per row is what made this figure noisy.
    # It is the T = 0.3 distribution only: rejection_distribution.csv is
    # per-scaffold and target-agnostic, and stacking three bands per row would
    # cost more than the widening is worth when the markers already carry it.
    ax.barh(ypos, 2 * 1.96 * dist["q_sd"], left=dist["q_mean"] - 1.96 * dist["q_sd"],
            height=0.55, color=colour, alpha=0.22, linewidth=0, zorder=1,
            label=f"vanilla sample mean ± 1.96 sd\n(T = {BASE_TEMPERATURE:g}, "
                  f"{n_label} samples)")
    ax.scatter(dist["q_mean"], ypos, marker="|", s=90, color=colour,
               linewidths=1.6, zorder=3)

    by_scaffold = {s: g for s, g in cells.groupby("scaffold_id")}
    for y, scaffold in zip(ypos, dist["scaffold_id"]):
        targets = by_scaffold.get(scaffold)
        if targets is None:
            continue
        for temperature, edge, _ in tiers:
            got = targets[np.isclose(targets["reached_at"], temperature)]
            ax.scatter(got["target_charge"], np.full(len(got), y), marker="o",
                       s=13, facecolor="white", edgecolor=edge, linewidths=1.0,
                       zorder=4)
        miss = targets[targets["reached_at"].isna()]["target_charge"]
        ax.scatter(miss, np.full(len(miss), y), marker="x", s=16,
                   color=REACH_MISS_COLOR, linewidths=1.0, zorder=4)

    ax.scatter(dist["q_wt"], ypos, marker="D", s=11, facecolor=TEXT,
               edgecolor=TEXT, zorder=5)

    # Proxy handles, so the legend swatches match what is actually drawn. The
    # old version drew the wild type white-faced and legended it in the pool
    # colour.
    for _, edge, label in tiers:
        ax.scatter([], [], marker="o", s=13, facecolor="white", edgecolor=edge,
                   linewidths=1.0, label=label)
    ax.scatter([], [], marker="x", s=16, color=REACH_MISS_COLOR, linewidths=1.0,
               label=f"not reached at T ≤ {max(have):g}")
    ax.scatter([], [], marker="D", s=11, color=TEXT, label="wild-type net charge")

    ax.set_yticks(ypos)
    ax.set_yticklabels(dist["scaffold_id"], fontsize=9)
    ax.set_ylim(-0.8, len(dist) - 0.2)
    ax.set_xlabel("net charge")
    ax.set_title("F1  The charge ladder against the unguided sampling distribution")
    ax.grid(axis="y", visible=False)
    _legend_right(ax)
    fig.text(0.01, -0.02,
             f"Unguided pools of {n_label} samples per scaffold at T = {t_label}. "
             "A marker states the lowest temperature whose pool landed exactly on "
             "that rung.", fontsize=9, color=MUTED, ha="left")
    fig.tight_layout()
    return fig


def reach_summary(data: dict) -> pd.DataFrame:
    """Cells reached per rung per temperature, the counts F1's notes quote."""
    cells, have = reach_tiers(data["curve_by_t"])
    rungs = (data["curve_by_t"][["scaffold_id", "target_charge", "delta_q_density"]]
             .drop_duplicates())
    cells = cells.merge(rungs, on=["scaffold_id", "target_charge"], how="left")
    rows = []
    for rung, group in cells.groupby("delta_q_density"):
        row = {"delta_q_density": rung, "n_scaffolds": len(group)}
        for t in have:
            row[f"reached_by_T{t:g}"] = int((group["reached_at"] <= t + 1e-9).sum())
        row["never_reached"] = int(group["reached_at"].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("delta_q_density")


# ---------------------------------------------------------------------------
# F2  mutational efficiency
#
# Reworked: fixed biological facet order plus a pooled facet, all nine arms, and
# an unguided-MPNN series from the Phase 4a pools.
# ---------------------------------------------------------------------------

def unguided_mut_per_charge(data: dict, fold: str | None) -> np.ndarray:
    """Unguided mutations per unit charge for one fold class, or all if None.

    No figure draws this any more; see fig2's docstring for why. It stays
    because the measurement stays: `unguided_by_fold_class` reports it in the
    notebook so the number is available without the series distorting an axis.

    The pools are per scaffold, not per fold class, so the fold class comes from
    the manifest. eGFP has none and therefore contributes only to the pooled
    facet, which is the same rule the guided arms follow here.
    """
    hits = data.get("hits")
    if hits is None or hits.empty:
        return np.array([])
    vals = hits.dropna(subset=["mut_per_charge"])
    if fold is not None:
        klass = data["manifest"].set_index("scaffold_id")["fold_class"]
        keep = klass[klass == fold].index
        vals = vals[vals["scaffold_id"].isin(keep)]
    return vals["mut_per_charge"].values


def fig2(data: dict):
    """Mutations per unit charge, guided arms only.

    The unguided rejection pool is deliberately NOT drawn here, though
    `results/rejection_hits.csv` still carries it and `unguided_summary` still
    reports it. It spends 8 to 12 mutations per unit charge against roughly 0.8
    for every guided arm, so including it forced a log axis on which the
    difference between the guided arms and the two Der et al. reference lines,
    which is what this figure is for, became unreadable. The unguided sampler's
    behaviour is already the whole subject of the charge-distribution figure
    that opens the notebook; it does not need restating on a linear axis it
    would dominate.
    """
    d = with_fold_class(data["designs"]).dropna(subset=["mut_per_charge"])
    order = methods_present(d)
    facets = FOLD_FACETS + [POOLED_FACET]
    series = order

    fig, axes = plt.subplots(1, len(facets), figsize=(3.4 * len(facets), 4.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fold in zip(axes, facets):
        pooled = fold == POOLED_FACET
        sub = d if pooled else d[d["fold_class"] == fold]
        boxes = [sub[sub["method"] == m]["mut_per_charge"].values for m in order]
        bp = ax.boxplot(boxes, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": TEXT, "linewidth": 1.4})
        _style_boxes(bp, series)
        for value, ls in ((DER_MUT_PER_CHARGE["AvNAPSA"], ":"),
                          (DER_MUT_PER_CHARGE["Rosetta"], "--")):
            ax.axhline(value, color=MUTED, ls=ls, lw=1.0)
        ax.set_xticks(range(1, len(series) + 1))
        ax.set_xticklabels(_tick_labels(series), rotation=45, ha="right",
                           fontsize=9)
        n_scaffolds = sub["scaffold_id"].nunique()
        ax.set_title(f"{fold.replace('_', ' ')}  (n = {n_scaffolds})", fontsize=9)

    axes[0].set_ylabel("mutations per unit charge moved")
    # The Der lines went unlabelled except by a text block pinned to the last
    # facet. They get real legend handles instead.
    handles = [plt.Line2D([], [], color=MUTED, ls=":", lw=1.0,
                          label="Der et al. AvNAPSA, 0.6"),
               plt.Line2D([], [], color=MUTED, ls="--", lw=1.0,
                          label="Der et al. Rosetta, 0.85")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=10,
                    loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("F2  Mutational efficiency", y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F3  sequence diversity
# ---------------------------------------------------------------------------

def fig3(data: dict):
    s = data["summary"]
    panels = [("n_unique_mutation_sets", "unique mutation sets (of n)"),
              ("mean_pairwise_hamming", "mean pairwise Hamming\n(designable positions)"),
              ("positional_entropy", "positional entropy (bits)"),
              ("designable_coverage", "designable coverage")]
    order = methods_present(s)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
    for ax, (col, label) in zip(axes, panels):
        boxes = [s[s["method"] == m][col].dropna().values for m in order]
        bp = ax.boxplot(boxes, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": TEXT, "linewidth": 1.4})
        _style_boxes(bp, order)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(_tick_labels(order), rotation=45, ha="right",
                           fontsize=9)
        ax.set_ylabel(label, fontsize=12)
    fig.suptitle("F3  Sequence diversity per scaffold × target cell",
                 y=1.03, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F4  delta REU per residue
# ---------------------------------------------------------------------------

def fig4(data: dict):
    d = data["designs"].dropna(subset=["d_reu_per_res"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for method in methods_present(d):
        m = d[d["method"] == method]
        grp = m.groupby("delta_q_density")["d_reu_per_res"]
        med, lo, hi = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
        st = _style_for(method)
        ax.plot(med.index, med.values, ms=4, **st)
        ax.fill_between(med.index, lo.values, hi.values, color=st["color"],
                        alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("ΔQ density (charge / 100 aa)")
    ax.set_ylabel("Δ REU per residue vs relaxed WT")
    ax.set_title("F4  Energetic cost")
    _legend_right(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F5  per-term decomposition, styled after Der et al. Figure 5
# ---------------------------------------------------------------------------

def fig5(data: dict):
    d = data["designs"]
    terms = ["d_fa_atr", "d_fa_rep", "d_fa_sol", "d_fa_elec",
             "d_hbond_sc", "d_hbond_bb_sc"]
    have = [t for t in terms if t in d.columns and d[t].notna().any()]
    if not have:
        return None
    order = methods_present(d.dropna(subset=have[:1]))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    width = 0.8 / max(len(order), 1)
    x = np.arange(len(have))
    for i, method in enumerate(order):
        m = d[d["method"] == method]
        vals = [m[t].median() for t in have]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
               color=METHOD_COLOR[method], label=METHOD_LABEL[method],
               edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("d_", "") for t in have])
    ax.set_ylabel("median Δ term per residue")
    ax.set_title("F5  Per-term energy decomposition")
    _legend_right(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F6  hydrogen bonds, with the protection on/off comparison made matched
#
# `rosetta` protects h-bonded sidechains and the primary MPNN arm does not, so
# the old two-panel version compared arms that differed on more than the design
# method. The added `rosetta_hbond_off` arm closes that, and the third panel
# puts the four series on the one scaffold set they share.
# ---------------------------------------------------------------------------

HBOND_PAIRS = [("mpnn_soluble", "mpnn_soluble_hbond_protected"),
               ("rosetta", "rosetta_hbond_off")]


def hbond_control_scaffolds(d: pd.DataFrame) -> list[str]:
    """Scaffolds carried by every arm in the protection comparison.

    Intersected rather than assumed: the two control arms are configured to the
    same subset, but a cell that failed on one and not the other would leave the
    panel comparing different scaffold sets without saying so.
    """
    sets = []
    for a, b in HBOND_PAIRS:
        for arm in (a, b):
            sets.append(set(d[d["method"] == arm]["scaffold_id"].unique()))
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


def fig6(data: dict):
    d = data["designs"]
    if "d_hbond_strong" not in d.columns or d["d_hbond_strong"].isna().all():
        return None
    d = d.dropna(subset=["d_hbond_strong"])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), sharex=True)
    for ax, col, title in ((axes[0], "d_hbond_strong", "strong (≤ −0.5 REU)"),
                           (axes[1], "d_hbond_weak", "weak (−0.5 to −0.1 REU)")):
        if col not in d.columns:
            continue
        for method in methods_present(d):
            m = d[d["method"] == method]
            med = m.groupby("delta_q_density")[col].median()
            st = _style_for(method)
            ax.plot(med.index, med.values, ms=4, **st)
        ax.axhline(0, color=MUTED, lw=1.0, ls="--")
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("ΔQ density")
    axes[0].set_ylabel("Δ side-chain H-bonds vs relaxed WT")

    ax = axes[2]
    shared = hbond_control_scaffolds(d)
    sub = d[d["scaffold_id"].isin(shared)]
    drawn = False
    for a, b in HBOND_PAIRS:
        for arm, ls in ((a, "-"), (b, ":")):
            m = sub[sub["method"] == arm]
            if m.empty:
                continue
            med = m.groupby("delta_q_density")["d_hbond_strong"].median()
            ax.plot(med.index, med.values, ls=ls, ms=4, **_style_for(arm))
            drawn = True
    if drawn:
        ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    else:
        ax.text(0.5, 0.5, "no scaffold is threaded on all four arms",
                ha="center", va="center", transform=ax.transAxes,
                color=MUTED, fontsize=10)
    ax.set_xlabel("ΔQ density")
    ax.set_title(f"h-bond protection on (dotted) vs off (solid), "
                 f"{len(shared)} shared scaffolds", fontsize=12, loc="left")
    # Panel c reuses the arm colours from the figure legend, so the only thing
    # it has to explain on its own is the linestyle convention.
    ax.legend(handles=[plt.Line2D([], [], color=MUTED, ls="-", lw=1.4,
                                  label="protection off"),
                       plt.Line2D([], [], color=MUTED, ls=":", lw=1.4,
                                  label="protection on")],
              frameon=False, fontsize=10, loc="upper left")

    # One legend for all nine arms at the figure foot. The first two panels draw
    # every arm and previously had no key at all: the only legend on this figure
    # sat on panel c and named four of them.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("F6  Hydrogen bonds gained and lost", y=1.03, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F7 structure prediction
#
# AF3 removed: 32 designs across 4 of 9 arms carried the overlay, and T4 in the
# original notebook remains the AF3 artifact. Third row added for the absolute
# pLDDT, so the figure reads confidence, change in confidence, and backbone
# deviation. Faceted rather than overlaid, because a scatter-shaped form puts
# arbitrary pairs of series side by side and the palette does not certify that.
# ---------------------------------------------------------------------------

F10_ROWS = [("plddt_mean", "ESMFold pLDDT"),
            ("d_plddt_vs_wt_pred", "Δ pLDDT vs WT prediction"),
            ("ca_rmsd_to_wt_crystal", "CA RMSD to WT crystal (Å)")]


def fig7(data: dict):
    d = data["designs"]
    if d["plddt_mean"].isna().all():
        return None
    d = d.dropna(subset=["plddt_mean"])
    order = methods_present(d)
    fig, axes = plt.subplots(len(F10_ROWS), len(order),
                             figsize=(1.9 * len(order), 6.6),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for j, method in enumerate(order):
        m = d[d["method"] == method]
        colour = METHOD_COLOR[method]
        for i, (col, _) in enumerate(F10_ROWS):
            ax = axes[i, j]
            if col not in m.columns or m[col].isna().all():
                continue
            grp = m.groupby("delta_q_density")[col]
            med, lo, hi = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
            ax.plot(med.index, med.values, color=colour,
                    marker=METHOD_MARKER[method], ms=3.5, lw=1.4)
            ax.fill_between(med.index, lo.values, hi.values, color=colour,
                            alpha=0.15, lw=0)
            if col == "d_plddt_vs_wt_pred":
                ax.axhline(0, color=MUTED, lw=1.0, ls="--")
            if col == "plddt_mean":
                ax.axhline(A.PLDDT_ABS_THRESHOLD, color=MUTED, lw=1.0, ls=":")
        axes[0, j].set_title(METHOD_LABEL[method].replace(" (", "\n("),
                             fontsize=10)
    for i, (_, label) in enumerate(F10_ROWS):
        axes[i, 0].set_ylabel(label, fontsize=12)
    for ax in axes[-1]:
        ax.set_xlabel("ΔQ density", fontsize=12)
    fig.suptitle("F7  Independent structure prediction (ESMFold2)",
                 y=1.01, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F8 runtime per accepted design
#
# Gains the unguided sampler, whose cost per on-target design is what
# rejection_curve.csv's expected_gpu_seconds_per_hit holds. That column name is
# legacy: it carries CPU seconds, per PLAN.md Section 0.2 D.
# ---------------------------------------------------------------------------

def unguided_cost_per_hit(curve: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Seconds per on-target unguided design, and how many cells are censored.

    A cell where the pool never hit has the string ">N" in this column, which is
    a lower bound, not a value. Those cells are excluded, exactly as the guided
    arms' zero-hit cells already are. Both exclusions push the same way and the
    count is returned so the notebook can say so.
    """
    raw = curve["expected_gpu_seconds_per_hit"].astype(str)
    censored = raw.str.startswith(">")
    vals = pd.to_numeric(raw[~censored], errors="coerce").dropna()
    return vals.values, int(censored.sum()), len(raw)


def fig8(data: dict):
    d = data["designs"]
    order = methods_present(d)
    per_cell = (d.groupby(["method", "scaffold_id", "target_charge"])
                .agg(wall=("wall_seconds", "first"),
                     hits=("hit_exact", "sum")).reset_index())
    per_cell = per_cell[per_cell["hits"] > 0]
    per_cell["cost"] = per_cell["wall"] / per_cell["hits"]

    boxes = [per_cell[per_cell["method"] == m]["cost"].dropna().values
             for m in order]
    series = list(order)
    curve = data.get("curve")
    if curve is not None and "expected_gpu_seconds_per_hit" in curve.columns:
        vals, _, _ = unguided_cost_per_hit(curve)
        boxes.append(vals)
        series.append(VANILLA)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bp = ax.boxplot(boxes, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={"color": TEXT, "linewidth": 1.4})
    _style_boxes(bp, series)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels(_tick_labels(series), rotation=45, ha="right",
                           fontsize=9)
    ax.set_ylabel("CPU-seconds per on-target design (log)")
    ax.set_title("F8  Runtime per accepted design")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F9 where predicted confidence starts to fall
#
# Reduced from four panels to two. The |dQ density| panel said the same thing as
# the mutation-count panel with a coarser axis, and the temperature panel is
# promoted to F10, where it sits next to the charge demand that drove the
# temperature up in the first place.
#
# The balancing helpers come from analysis_lib: plddt_frame reconstructs each
# scaffold's predicted wild type by subtraction, since no column stores it, and
# scaffold_balanced takes per-scaffold medians first so an unbalanced axis
# cannot report the scaffold panel instead of the effect.
# ---------------------------------------------------------------------------

# Panel d's below-threshold highlight. analysis_lib used the AvNAPSA method
# colour here, which read as a method reference directly above a method legend.
# Neutral by design: this bar chart has no methods in it.
BELOW_THRESHOLD = "#c1272d"

# The mutation axis stops here. analysis_lib.MUT_BIN_LABELS runs on to "41-60"
# and ">60", but no arm puts 5 scaffolds in those bins, so they were drawn as
# three empty tick positions carrying a fifth of the axis width and nothing
# else. Truncating is not hiding data: the bins the rule drops are named in the
# notes and the rule itself is unchanged.
MUT_BIN_LAST = "31-40"
MUT_BIN_LABELS_SHOWN = A.MUT_BIN_LABELS[:A.MUT_BIN_LABELS.index(MUT_BIN_LAST) + 1]


def _plot_balanced(ax, frame, by, method, col="d_plddt_vs_wt_pred", x_of=None,
                   label=None, ls="-"):
    bal = A.scaffold_balanced(frame, by, col)
    if bal.empty:
        return False
    x = [x_of(v) for v in bal.index] if x_of else list(bal.index)
    ax.plot(x, bal["median"].values, color=METHOD_COLOR.get(method, MUTED),
            marker=METHOD_MARKER.get(method, "o"), ms=4, lw=1.4, ls=ls,
            label=label or METHOD_LABEL.get(method, method))
    ax.fill_between(x, bal["q25"].values, bal["q75"].values,
                    color=METHOD_COLOR.get(method, MUTED), alpha=0.12, lw=0)
    return True


def fig9(data: dict):
    d = A.plddt_frame(data)
    if d.empty:
        return None
    order = methods_present(d)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    for method in order:
        _plot_balanced(ax, d[d["method"] == method], "mut_bin", method,
                       x_of=lambda v: A.MUT_BIN_LABELS.index(str(v)))
    A._reference_lines(ax)
    ax.set_xticks(range(len(MUT_BIN_LABELS_SHOWN)))
    ax.set_xticklabels(MUT_BIN_LABELS_SHOWN, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-0.4, len(MUT_BIN_LABELS_SHOWN) - 0.6)
    ax.set_xlabel("mutations")
    ax.set_ylabel("Δ pLDDT vs WT prediction")
    ax.set_title("a  against mutation count", fontsize=12, loc="left")

    ax = axes[1]
    wt = A.wt_below_threshold(d)
    colours = [BELOW_THRESHOLD if v < A.PLDDT_ABS_THRESHOLD else MUTED
               for v in wt.values]
    ax.bar(range(len(wt)), wt.values, color=colours, width=0.8)
    ax.axhline(A.PLDDT_ABS_THRESHOLD, color=TEXT, lw=1.2, ls="--")
    n_below = int((wt < A.PLDDT_ABS_THRESHOLD).sum())
    ax.annotate(f"{n_below} of {len(wt)} wild types already below "
                f"{A.PLDDT_ABS_THRESHOLD:.0f}",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=10,
                color=TEXT)
    ax.set_xticks(range(len(wt)))
    ax.set_xticklabels(wt.index, rotation=90, fontsize=8)
    ax.set_ylabel("predicted WT pLDDT")
    ax.set_title("b  why the absolute cutoff is scaffold-dependent",
                 fontsize=12, loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("F9  Where predicted confidence starts to fall", y=1.02,
                 fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F10 charge demand, decoding temperature, predicted confidence
#
# Added 2026-09-05, PLAN.md Section 11.1. final_temperature exists only on the
# MPNN arms: it is written by the `-u` escalation loop, which raises T in 0.1
# steps from 0.3 to a 0.9 ceiling until the target is hit. The classical arms
# and the random control have no such knob and the column is blank for them.
#
# Read left to right: how much temperature a target cost, then what that
# temperature coincided with in predicted confidence, relative and absolute.
# "Coincided with" is deliberate. Temperature and mutation count rise together
# here, so neither of the right-hand panels isolates temperature, and the
# notebook says so rather than the figure implying otherwise.
#
# POOLED is drawn beside the per-arm lines because the escalation is rare: most
# cells never leave 0.3, so the high-T bins hold few scaffolds per arm and the
# 5-scaffold rule terminates the per-arm lines early. The pooled line is every
# MPNN design at once and reaches further up the axis.
# ---------------------------------------------------------------------------

POOLED = "__pooled_mpnn__"
METHOD_COLOR[POOLED] = TEXT
METHOD_MARKER[POOLED] = "o"
METHOD_LABEL[POOLED] = "all MPNN arms pooled"


def temperature_frame(data: dict) -> pd.DataFrame:
    """MPNN rows with a numeric final_temperature and a WT pLDDT reference.

    `final_temperature` arrives as mixed str/float because the baseline arms
    leave it blank and pandas types the whole column from the whole frame, so it
    is coerced once here rather than at each use, the same way fig11 does it.
    """
    d = A.plddt_frame(data)
    d = d[d["method"].isin(MPNN_ARMS)].copy()
    d["final_temperature"] = pd.to_numeric(d["final_temperature"],
                                           errors="coerce")
    return d.dropna(subset=["final_temperature"])


def fig10(data: dict):
    d = temperature_frame(data)
    if d.empty:
        return None
    order = [m for m in methods_present(d) if m in MPNN_ARMS]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    # a. what the target cost in temperature
    #
    # Signed, not |ΔQ density|, on the author's instruction 2026-09-06 (PLAN.md
    # Section 0.1 item 24). The fold put -4 on top of +4 and so on, showing the
    # eight-rung ladder at four x positions and averaging the two supercharging
    # directions together. They are not symmetric: the unguided pools reach -24
    # on every scaffold and +24 on none, and temperature_confound puts the median
    # ΔpLDDT at T = 0.9 at -6.60 going negative against -0.36 going positive, so
    # the fold was averaging over the thing worth seeing. Doubling the bins costs
    # nothing here: temperature_bin_counts shows all eight rungs hold every
    # scaffold the arm ran, so A.scaffold_balanced's 5-scaffold rule drops none
    # of them and no line stops short.
    ax = axes[0]
    for method in order:
        _plot_balanced(ax, d[d["method"] == method], "delta_q_density", method,
                       col="final_temperature")
    ax.axhline(0.3, color=MUTED, lw=1.0, ls="--")
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_xticks(sorted(d["delta_q_density"].unique()))
    ax.set_xlabel("ΔQ density")
    ax.set_ylabel("final decoding temperature")
    ax.set_title("a  temperature the target cost", fontsize=12, loc="left")

    # b and c. what that coincided with in predicted confidence
    for ax, col, ylabel, title in (
            (axes[1], "d_plddt_vs_wt_pred", "Δ pLDDT vs WT prediction",
             "b  change in confidence"),
            (axes[2], "plddt_mean", "ESMFold pLDDT",
             "c  absolute confidence")):
        for method in order:
            _plot_balanced(ax, d[d["method"] == method], "final_temperature",
                           method, col=col)
        _plot_balanced(ax, d, "final_temperature", POOLED, col=col, ls="--")
        if col == "d_plddt_vs_wt_pred":
            A._reference_lines(ax)
        else:
            ax.axhline(A.PLDDT_ABS_THRESHOLD, color=TEXT, lw=1.0, ls="--")
        ax.set_xlabel("final decoding temperature")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, loc="left")

    _legend_right(axes[2])
    fig.suptitle("F10  Charge demand, decoding temperature, predicted "
                 "confidence (MPNN arms)", y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F11 charge capacity ceiling
#
# Restored 2026-09-06 on the author's instruction (PLAN.md Section 0.1 item 26).
# It was one of the six-arm figures retired the same day, and it is the only one
# whose question nothing else answers: how much charge the method can actually
# move on a given scaffold. F1 is the complement, what unguided sampling reaches
# without the method, and the two are read together.
#
# PLAN.md Section 11.1 calls this manuscript Figure 2A generalised from one
# protein to the panel, so it is the multi-scaffold form of the eGFP range test.
#
# **The ceiling is censored for most scaffolds and the figure says so.** The
# ladder stops at |ΔQ density| 24, so a scaffold that reaches 24 has reached the
# largest target it was ever asked for, not its limit. Twenty of 25 scaffolds
# are in that state on the primary arm. Those points are drawn as upward arrows
# meaning "at least 24" and are excluded from the medians, because a median over
# a mostly censored sample would report the ladder rather than the method. The
# uncensored minority is what the medians describe, and the panel title says how
# many that is. Reading a ceiling of 24 off this figure is reading the cap.
#
# Decision E (Section 0.1 item 9) fixes the definition: the ceiling is the
# largest |ΔQ density| hit with final_temperature == 0.3, so it is capacity at
# the base decoding temperature rather than capacity after escalation.
# ---------------------------------------------------------------------------

LADDER_CAP = 24


def capacity_ceiling(data: dict, method: str = "mpnn_soluble") -> pd.Series:
    """Largest |ΔQ density| an arm hit at T = 0.3, per scaffold.

    Returns an empty series rather than raising if the arm never hit at 0.3, so
    a caller can report the absence instead of inventing a ceiling.
    """
    d = data["designs"]
    arm = d[d["method"] == method].copy()
    # final_temperature is blank on the non-MPNN arms, so the column arrives as
    # mixed str/float and any ordered comparison on it raises. Coerce once.
    arm["final_temperature"] = pd.to_numeric(arm["final_temperature"],
                                             errors="coerce")
    at_base = arm[arm["hit_exact"] & (arm["final_temperature"] == 0.3)]
    if at_base.empty:
        return pd.Series(dtype=float)
    return (at_base.assign(a=at_base["delta_q_density"].abs())
            .groupby("scaffold_id")["a"].max())


def fig11(data: dict):
    ceiling = capacity_ceiling(data)
    if ceiling.empty:
        return None
    manifest = data["manifest"].set_index("scaffold_id")
    colour = METHOD_COLOR["mpnn_soluble"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    # a. capacity by fold class. eGFP has no fold class, having been added
    # outside the CATH split, and is labelled rather than dropped: it is the
    # scaffold manuscript Figure 2A is about and this figure generalises it.
    classes = manifest.loc[ceiling.index, "fold_class"].fillna(A.FOCUS_CLASS_LABEL)
    names = [f for f in FOLD_FACETS if f in set(classes)] + \
            [f for f in sorted(set(classes)) if f not in FOLD_FACETS]
    rng = np.random.default_rng(0)
    ax = axes[0]
    n_censored = 0
    for i, fold in enumerate(names):
        vals = ceiling[classes == fold]
        at_cap = vals >= LADDER_CAP
        n_censored += int(at_cap.sum())
        x = i + rng.uniform(-0.13, 0.13, len(vals))
        free_x, cap_x = x[~at_cap.values], x[at_cap.values]
        # Below the cap: a measured ceiling, filled.
        ax.scatter(free_x, vals[~at_cap], s=42, color=colour, alpha=0.85,
                   zorder=3)
        # At the cap: censored, so drawn as "at least", open and arrow-headed.
        ax.scatter(cap_x, vals[at_cap], s=60, marker="^", facecolor="white",
                   edgecolor=colour, linewidths=1.3, zorder=3)
        free = vals[~at_cap]
        if len(free):
            ax.hlines(free.median(), i - 0.26, i + 0.26, color=TEXT, lw=2,
                      zorder=4)
    ax.axhline(LADDER_CAP, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.annotate(f"ladder cap, |ΔQ density| {LADDER_CAP}", xy=(0.02, LADDER_CAP),
                xycoords=("axes fraction", "data"), va="bottom", fontsize=9,
                color=MUTED)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20,
                       ha="right")
    ax.set_ylabel("max |ΔQ density| reached at T = 0.3")
    ax.set_ylim(0, LADDER_CAP * 1.18)
    ax.set_title(f"a  per-scaffold charge capacity  "
                 f"({n_censored} of {len(ceiling)} censored at the cap)",
                 fontsize=12, loc="left")
    ax.scatter([], [], s=42, color=colour, label="ceiling measured")
    ax.scatter([], [], s=60, marker="^", facecolor="white", edgecolor=colour,
               linewidths=1.3, label=f"≥ {LADDER_CAP}, never asked for more")
    ax.hlines([], [], [], color=TEXT, lw=2, label="median of the measured")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # b. what each cell cost in temperature. Signed rungs, so the direction
    # asymmetry F10 quantifies is visible on the scaffold grid too.
    arm = data["designs"]
    arm = arm[arm["method"] == "mpnn_soluble"].copy()
    arm["final_temperature"] = pd.to_numeric(arm["final_temperature"],
                                             errors="coerce")
    pivot = arm.pivot_table(index="scaffold_id", columns="delta_q_density",
                            values="final_temperature", aggfunc="max")
    pivot = pivot.loc[ceiling.sort_values(kind="stable").index]
    ax = axes[1]
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("ΔQ density")
    ax.set_title("b  final decoding temperature the cell required", fontsize=12,
                 loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="final temperature")

    fig.suptitle("F11  Charge capacity ceiling (MPNN supercharge, soluble), "
                 f"manuscript Fig. 2A generalised to {len(ceiling)} scaffolds",
                 y=1.03, fontsize=14)
    fig.tight_layout()
    return fig


def capacity_by_arm(data: dict) -> pd.DataFrame:
    """Ceiling per MPNN arm, with the censored count stated rather than buried.

    F11 draws the primary arm. This is the same measurement for all five, and it
    is where the notes cell reads its numbers from.
    """
    rows = []
    for method in MPNN_ARMS:
        c = capacity_ceiling(data, method)
        if c.empty:
            continue
        at_cap = c >= LADDER_CAP
        free = c[~at_cap]
        rows.append({
            "method": METHOD_LABEL[method],
            "n_scaffolds": len(c),
            "n_at_ladder_cap": int(at_cap.sum()),
            "median_all_censored": c.median(),
            "median_measured_only": free.median() if len(free) else float("nan"),
            "min": c.min(),
        })
    return pd.DataFrame(rows).set_index("method")

# ---------------------------------------------------------------------------
# numbers the notebook's notes cells quote, computed rather than typed
# ---------------------------------------------------------------------------

def temperature_confound(data: dict) -> pd.DataFrame:
    """Median mutation count and scaffold coverage per temperature bin.

    F10's notes claim temperature and mutation count are confounded and that the
    high-T bins are thin. Both claims are read off this frame in the notebook so
    neither is a number somebody typed.

    Split by supercharging direction as well as by temperature, so the asymmetry
    F10 panel a now draws is available as a number. `direction` is the sign of
    `delta_q_density`, which is what chooses the direction in the runs
    themselves; the ladder has no zero rung, so there is no third group.
    """
    d = temperature_frame(data)
    d = d.assign(direction=np.where(d["delta_q_density"] > 0,
                                    "supercharge +", "supercharge -"))
    out = d.groupby(["direction", "final_temperature"]).agg(
        n_designs=("n_mutations", "size"),
        n_scaffolds=("scaffold_id", "nunique"),
        median_n_mutations=("n_mutations", "median"),
        median_plddt=("plddt_mean", "median"),
        median_d_plddt=("d_plddt_vs_wt_pred", "median"))
    return out.round(2)


def temperature_bin_counts(data: dict) -> pd.DataFrame:
    """Scaffolds per (arm, signed rung), the counts F10 panel a's notes quote.

    A.scaffold_balanced drops any bin holding fewer than MIN_SCAFFOLDS_PER_BIN
    scaffolds, so signing panel a's x axis could have thinned it. This is the
    check that says whether it did, rather than the figure being read as if the
    question had not come up.
    """
    d = temperature_frame(data)
    out = (d[d["method"].isin(MPNN_ARMS)]
           .groupby(["method", "delta_q_density"])["scaffold_id"].nunique()
           .unstack("delta_q_density"))
    out.columns.name = "delta_q_density"
    return out


def unguided_by_fold_class(data: dict) -> pd.DataFrame:
    """Unguided mutations per unit charge, per fold class and pooled.

    The counterpart to the guided arms in the mutational-efficiency figure,
    reported as a table rather than drawn, because on a shared axis it is an
    order of magnitude away and flattens everything the figure compares.
    """
    rows = []
    for fold in FOLD_FACETS + [POOLED_FACET]:
        vals = unguided_mut_per_charge(data, None if fold == POOLED_FACET else fold)
        s = pd.Series(vals)
        rows.append({"fold_class": fold, "n_on_target_samples": len(s),
                     "median_mut_per_charge": round(s.median(), 3) if len(s) else None,
                     "q25": round(s.quantile(0.25), 3) if len(s) else None,
                     "q75": round(s.quantile(0.75), 3) if len(s) else None})
    return pd.DataFrame(rows)


def unguided_summary(data: dict) -> pd.DataFrame:
    """Per-rung coverage of the unguided series F2 and F8 draw."""
    curve = data["curve"]
    hits = data.get("hits")
    out = curve.groupby("delta_q_density").agg(
        cells=("n_on_target", "size"),
        cells_with_a_hit=("n_on_target", lambda s: int((s > 0).sum())),
        on_target_samples=("n_on_target", "sum"))
    if hits is not None and not hits.empty:
        med = hits.groupby("delta_q_density")["mut_per_charge"].median()
        out = out.join(med.rename("median_mut_per_charge"))
    return out.round(3)

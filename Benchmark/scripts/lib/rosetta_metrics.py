"""REU terms and hydrogen-bond counts for Phase 5.

PLAN.md Section 8. Everything here operates on a pose already written to disk by
`threading_only.py`, which keeps the lowest-energy structure across its
`--num_relax` restarts. Nothing here re-relaxes a design; the only relaxation
this module performs is the wild-type reference, for the reason below.

**The wild-type reference has to be built here, and it is not literally the
"identical protocol" Section 8 asks for.** `threading_only.py:234-239` short
circuits when a sequence carries no mutations:

    if not mutant_index_list:
        pose.dump_pdb(output_path)
        sfxn(pose)
        return total_energy(pose, sfxn), mutation_labels

so it dumps and scores the *unrelaxed crystal pose*. Every design, by contrast,
goes through FastRelax. Referencing relaxed designs against an unrelaxed wild
type would put the entire relaxation gain into `d_reu_per_res`: the crystal pose
of `1a1x_A` scores `fa_rep` 173.2 before relaxation, and essentially all of that
is removed by relax, so every design would appear enormously better than wild
type for reasons that have nothing to do with its mutations.

Section 8 says "relax the WT under the identical protocol once per scaffold per
tier". Taken literally the design protocol is undefined for a sequence with no
mutations, because it restricts the movemap and the task factory to the mutated
positions and their neighbour shell, and with zero mutations that shell is empty,
which is exactly the degenerate no-op above. `relax_wt_reference` therefore uses
the same score function (`get_fa_scorefxn()`, ref2015), the same
`fast_relax_mover` with the same `FastRelax(5)` repeats, and the same
lowest-of-`num_relax` retention, but with an unrestricted movemap and a
repack-everything task factory.

That is a documented deviation, not a silent one, and it is not neutral: the
wild type gets more conformational freedom than any design does, which biases
`d_reu_per_res` *against* the designs. It is the conservative direction, but it
is a bias, so `score_pdb` also records the unrelaxed crystal score of every
scaffold as `reu_total_crystal` and the per-cell record keeps both. The notebook
can then show how much of any effect depends on the choice.

`threading_only.py` is not modified. See logs/ISSUES.md.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen term lists
# ---------------------------------------------------------------------------

# PLAN.md Section 8 names these eight per-term deltas. The first six reach
# results/designs.csv under the frozen Section 10.1 names d_fa_atr, d_fa_rep,
# d_fa_sol, d_fa_elec, d_hbond_sc, d_hbond_bb_sc. fa_dun and p_aa_pp are
# computed and kept in the per-cell JSON but have no column in the frozen
# schema, so they do not reach designs.csv without a PLAN.md change first.
SCORE_TERMS = ["fa_atr", "fa_rep", "fa_sol", "fa_elec",
               "hbond_sc", "hbond_bb_sc", "fa_dun", "p_aa_pp"]

DESIGNS_CSV_TERMS = ["fa_atr", "fa_rep", "fa_sol", "fa_elec",
                     "hbond_sc", "hbond_bb_sc"]

# Der et al. 2013 thresholds, on the raw (unweighted) hbond energy that
# HBondSet reports. Section 8 specifies these numbers directly.
HBOND_STRONG_MAX = -0.5
HBOND_WEAK_MAX = -0.1


def init_pyrosetta(seed: int, extra: str = "") -> str:
    """Initialise PyRosetta with a recorded constant seed.

    Returns the option string actually used, for the per-cell record. `-run:jran`
    is parsed as a signed 32-bit integer, so the seed is folded the same way
    lib/baseline.py folds it rather than being narrowed at the source.
    """
    import pyrosetta                                               # noqa: PLC0415

    from lib.baseline import jran_for                              # noqa: PLC0415

    options = f"-run:constant_seed -constant_seed -mute all -jran {jran_for(seed)}"
    if extra:
        options = f"{options} {extra}"
    pyrosetta.init(options=options, silent=True)
    return options


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def hbond_counts(pose) -> dict:
    """Count side-chain hydrogen bonds by strength.

    Only bonds with at least one side-chain partner are counted; a bond that is
    backbone on both ends is excluded, because Section 8 is testing what the
    surface mutations did to side-chain hydrogen bonding and backbone-backbone
    bonds are a property of the fixed scaffold.

    Energies are the raw HBondSet energies, not multiplied by their score
    function weights, because the -0.5 and -0.1 thresholds come from Der et al.
    and are stated on the raw values.
    """
    from pyrosetta.rosetta.core.scoring.hbonds import (                # noqa: PLC0415
        HBondSet, fill_hbond_set)

    pose.update_residue_neighbors()
    hbset = HBondSet()
    fill_hbond_set(pose, False, hbset)

    strong = weak = sidechain = 0
    for i in range(1, hbset.nhbonds() + 1):
        hb = hbset.hbond(i)
        if hb.don_hatm_is_protein_backbone() and hb.acc_atm_is_protein_backbone():
            continue
        sidechain += 1
        energy = hb.energy()
        if energy <= HBOND_STRONG_MAX:
            strong += 1
        elif energy <= HBOND_WEAK_MAX:
            weak += 1
    return {"n_hbond_sidechain": sidechain,
            "n_hbond_strong": strong,
            "n_hbond_weak": weak}


def score_pose(pose, sfxn) -> dict:
    """Total REU, the eight per-term energies, and the hydrogen-bond counts."""
    from pyrosetta.rosetta.core import scoring                     # noqa: PLC0415

    sfxn(pose)
    totals = pose.energies().total_energies()
    out = {"reu_total": float(totals[scoring.ScoreType.total_score]),
           "n_residues": int(pose.total_residue())}
    for term in SCORE_TERMS:
        out[term] = float(totals[getattr(scoring, term)])
    out.update(hbond_counts(pose))
    return out


def score_pdb(pdb_path: str | Path, sfxn=None) -> dict:
    """Score a PDB already on disk. No relaxation, no repacking."""
    import pyrosetta                                               # noqa: PLC0415

    if sfxn is None:
        sfxn = pyrosetta.get_fa_scorefxn()
    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    record = score_pose(pose, sfxn)
    record["sequence"] = pose.sequence()
    return record


def relax_wt_reference(pdb_path: str | Path, out_pdb: str | Path,
                       num_relax: int) -> dict:
    """Relax the wild type and score it. See this module's docstring.

    Mirrors the design path's score function, FastRelax repeat count and
    lowest-of-`num_relax` retention, with an unrestricted movemap because the
    design path's shell is empty for a sequence with no mutations.

    Returns the score record for the retained pose, plus the unrelaxed crystal
    score under `reu_total_crystal` so the size of the relaxation gain stays
    visible in the record rather than being folded silently into every delta.
    """
    import pyrosetta                                               # noqa: PLC0415
    from pyrosetta import MoveMap, Pose                            # noqa: PLC0415
    from pyrosetta.rosetta.core.pack.task import TaskFactory       # noqa: PLC0415
    import pyrosetta.rosetta.core.pack.task.operation as taskop    # noqa: PLC0415
    from pyrosetta.rosetta.protocols.relax import FastRelax        # noqa: PLC0415

    sfxn = pyrosetta.get_fa_scorefxn()
    crystal = pyrosetta.pose_from_pdb(str(pdb_path))
    sfxn(crystal)
    crystal_total = float(
        crystal.energies().total_energies()[
            pyrosetta.rosetta.core.scoring.ScoreType.total_score])

    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)

    # Repack everything, design nothing. IncludeCurrent and NoRepackDisulfides
    # match make_task_factory in threading_only.py; ExtraRotamers is omitted
    # there for the mutant path (ex12=False) so it is omitted here too.
    tf = TaskFactory()
    tf.push_back(taskop.IncludeCurrent())
    tf.push_back(taskop.NoRepackDisulfides())
    tf.push_back(taskop.RestrictToRepacking())

    best_pose, best_energy = None, float("inf")
    for _ in range(num_relax):
        work = Pose()
        work.detached_copy(crystal)
        # FastRelax(5): the same repeat count fast_relax_mover uses by default.
        relax = FastRelax(5)
        relax.set_scorefxn(sfxn)
        relax.set_task_factory(tf)
        relax.set_movemap(movemap)
        relax.apply(work)
        energy = float(
            work.energies().total_energies()[
                pyrosetta.rosetta.core.scoring.ScoreType.total_score])
        if energy < best_energy:
            best_energy, best_pose = energy, work

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    best_pose.dump_pdb(str(out_pdb))

    record = score_pose(best_pose, sfxn)
    record["sequence"] = best_pose.sequence()
    record["reu_total_crystal"] = crystal_total
    record["reu_relaxation_gain"] = crystal_total - record["reu_total"]
    record["num_relax"] = num_relax
    record["wt_relax_protocol"] = "unrestricted_movemap_repack_all"
    return record


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------

def deltas_vs_wt(design: dict, wt: dict) -> dict:
    """Per-residue deltas of a design against its matched relaxed wild type.

    Section 8 normalises every REU quantity by residue count. Hydrogen-bond
    counts are integers and are reported as raw differences, not per residue,
    because a count per residue is not the quantity Der et al. Figure 6 plots.

    Raises on a residue-count mismatch rather than normalising by the wrong
    number: threading preserves the backbone, so a mismatch means the design and
    the reference are not the same scaffold.
    """
    if design["n_residues"] != wt["n_residues"]:
        raise ValueError(
            f"residue count mismatch: design {design['n_residues']}, "
            f"wt reference {wt['n_residues']}. These are not the same scaffold.")
    n = design["n_residues"]
    if n <= 0:
        raise ValueError(f"non-positive residue count {n}")

    out = {"d_reu_per_res": (design["reu_total"] - wt["reu_total"]) / n}
    for term in SCORE_TERMS:
        out[f"d_{term}"] = (design[term] - wt[term]) / n
    out["d_hbond_strong"] = design["n_hbond_strong"] - wt["n_hbond_strong"]
    out["d_hbond_weak"] = design["n_hbond_weak"] - wt["n_hbond_weak"]
    out["d_hbond_sidechain"] = design["n_hbond_sidechain"] - wt["n_hbond_sidechain"]
    return out

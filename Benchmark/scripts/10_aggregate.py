#!/usr/bin/env python3
"""Phase 7a: build results/designs.csv, scaffold_summary.csv and runtime.csv.

PLAN.md Section 10. The `designs.csv` schema in Section 10.1 is frozen and lives
in `lib/io.DESIGNS_COLUMNS`; this script fills it and validates every row against
it before writing, so a drift between the writer and the plan is an error rather
than a silently wider CSV.

Everything here is read from a file on disk. Nothing is interpolated, defaulted
to a plausible value, or carried over from another design. A metric that could
not be computed is left empty and, where the design itself failed, `status` is
`failed` with a `fail_reason`.

**One reference designable set per scaffold, shared by every arm.** The
diversity metrics in Section 10.2 and `frac_designable_mutated` are all
denominated in designable positions. The MPNN arms have a LayerSelector cache
that defines that set exactly; the AvNAPSA and Rosetta arms do not, because the
Supercharge mover picks its own surface internally, and the random control was
built against the MPNN set by construction. Using each arm's own notion of
"designable" would give every arm a different denominator and make the numbers
incomparable, which is the opposite of what Section 10.2 is for. The primary
arm's set is therefore used as the common frame for all arms, and for the two
classical baselines it is a reference set they did not themselves use. The
control arm is the one exception: it really does have a different designable set
(62 positions on eGFP against 96), so it reads its own cache.

**`wall_seconds` is the cell's, not the design's.** A cell is one invocation
that emits n designs, and the runtime is not separable per design. Every row of
a cell carries the same value, and `runtime.csv` is the per-job artifact Table
T5 should be built from.

Usage:
  10_aggregate.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import charge as charge_lib   # noqa: E402
from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402
from lib import metrics as met         # noqa: E402

SUMMARY_COLUMNS = [
    "scaffold_id", "pdb_id", "fold_class", "n_residues", "n_designable", "q_wt",
    "method", "weights", "hbond_filter", "delta_q_density", "target_charge",
    "n_designs", "n_hit_exact", "hit_rate",
    "mean_q_error_abs", "median_n_mutations", "median_mut_per_charge",
    "n_unique_mutation_sets", "mean_pairwise_hamming", "positional_entropy",
    "designable_coverage",
    "median_frac_designable_mutated", "median_seq_identity_to_wt_designable",
    "n_threaded", "median_d_reu_per_res", "median_d_hbond_strong",
    "median_d_hbond_weak",
    "n_predicted", "median_plddt_mean", "median_d_plddt_vs_wt_pred",
    "median_tm_score_to_wt_crystal", "median_ca_rmsd_to_wt_crystal",
    "cell_wall_seconds",
]

# Which working directory's LayerSelector cache defines a method's designable
# set. See the module docstring: only the control arm genuinely has its own.
CONTROL_ARM = "mpnn_soluble_hbond_protected"


def hbond_filter_for(cell: dict):
    """The h-bond protection state of one cell, PLAN.md Section 5 convention.

    True means h-bonded sidechains were PROTECTED. The MPNN arms record it
    directly, derived from the -mhbond flag by lib/io.hbond_filter_from_flag.

    The score-based Rosetta arms record it directly too, but only since
    2026-09-05: the 200 `rosetta` cells that ran before that wrote null, because
    the setting was hardcoded and lib/baseline.py had nothing to record. Their
    value is not unknown, it is written in the same sidecar under
    `mover_settings.dont_mutate_hbonded_sidechains`, which is what the mover was
    actually configured with. It is read from there rather than assumed, and no
    sidecar is rewritten: the recorded outputs of a completed run stay as they
    were written.

    AvNAPSA stays null. Its sequence-based surface definition never consults the
    switch, so there is no value to report and writing one would claim a setting
    the arm did not use.
    """
    value = cell.get("hbond_filter")
    if value is not None and value != "":
        return value
    return cell.get("mover_settings", {}).get("dont_mutate_hbonded_sidechains", "")


def log(msg: str) -> None:
    print(msg, flush=True)


def median(values: list) -> float | str:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return ""
    n = len(vals)
    mid = n // 2
    return float(vals[mid]) if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def mean(values: list) -> float | str:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else ""


# ---------------------------------------------------------------------------
# side tables
# ---------------------------------------------------------------------------

def load_threading() -> dict[tuple[str, int], dict]:
    """(cell_id, sample_index) -> the scored threading record for that design."""
    out: dict[tuple[str, int], dict] = {}
    root = config_lib.bench_path("results", "threaded")
    if not root.exists():
        return out
    for tier_dir in sorted(root.glob("*")):
        if not tier_dir.is_dir():
            continue
        for path in sorted(tier_dir.glob("*.json")):
            if path.name.startswith("_wt_"):
                continue
            with open(path) as fh:
                rec = json.load(fh)
            if rec.get("status") != "ok":
                continue
            for design in rec.get("designs", []):
                if design.get("status") != "ok":
                    continue
                design = dict(design)
                design["thread_tier"] = rec["tier"]
                out[(rec["cell_id"], int(design["sample_index"]))] = design
    return out


def load_csv_by(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as fh:
        return {row[key]: row for row in csv.DictReader(fh)}


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------

def build(cfg: dict, dry_run: bool) -> None:
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    designs_csv = config_lib.bench_path("results", "designs.csv")
    summary_csv = config_lib.bench_path("results", "scaffold_summary.csv")
    runtime_csv = config_lib.bench_path("results", "runtime.csv")
    esmfold_csv = config_lib.bench_path("results", "esmfold.csv")
    af3_csv = config_lib.bench_path("results", "af3.csv")

    if dry_run:
        print(f"[dry-run] would read {manifest_path}, every results/cells/*.json "
              "with status ok, and each cell's designs/*.fa")
        print(f"[dry-run] would join results/threaded/*/*.json "
              f"({'present' if config_lib.bench_path('results', 'threaded').exists() else 'NOT PRESENT'}), "
              f"{esmfold_csv.name} ({'present' if esmfold_csv.exists() else 'NOT PRESENT'}) and "
              f"{af3_csv.name} ({'present' if af3_csv.exists() else 'NOT PRESENT'})")
        print(f"[dry-run] would write {designs_csv} on the frozen "
              f"{len(bio.DESIGNS_COLUMNS)}-column schema, validating every row")
        print(f"[dry-run] would write {summary_csv} and {runtime_csv}")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Phase 1 must complete first.")
    with open(manifest_path) as fh:
        manifest = {r["scaffold_id"]: r for r in csv.DictReader(fh)}

    threading = load_threading()
    esmfold = load_csv_by(esmfold_csv, "record_id")
    af3 = load_csv_by(af3_csv, "design_id")

    # Designable sets. One read per (scaffold, workdir) rather than per cell.
    caches: dict[tuple[str, str], tuple] = {}

    def designable_for(scaffold_id: str, method: str) -> tuple[str, list[int]]:
        workdir = "data/hbond_protected" if method == CONTROL_ARM else "data"
        key = (scaffold_id, workdir)
        if key not in caches:
            pkl = config_lib.bench_path(workdir, "parsed", f"{scaffold_id}_seq_indices.pkl")
            wt_seq, indices, _ = bio.load_designable_cache(pkl)
            caches[key] = (wt_seq, indices)
        return caches[key]

    design_rows: list[dict] = []
    summary_rows: list[dict] = []
    runtime_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for path in sorted(config_lib.bench_path("results", "cells").glob("*.json")):
        with open(path) as fh:
            cell = json.load(fh)
        cell_id = cell["cell_id"]
        scaffold_id = cell["scaffold_id"]
        method = cell["method"]
        scaffold = manifest[scaffold_id]

        runtime_rows.append({
            "phase": "design", "cell_id": cell_id,
            "slurm_job_id": cell.get("slurm_job_id", ""),
            "slurm_array_task_id": cell.get("slurm_array_task_id", ""),
            "partition": cell.get("partition", ""), "node": cell.get("node", ""),
            "cpus": cell.get("cpus", ""), "gpus": cell.get("gpus", ""),
            "wall_seconds": cell.get("wall_seconds", ""),
            "status": cell.get("status", ""),
        })

        if cell.get("status") != "ok":
            skipped.append((cell_id, f"cell status={cell.get('status')}"))
            continue
        fasta = config_lib.bench_path("designs", f"{cell_id}.fa")
        if not fasta.exists():
            skipped.append((cell_id, "no FASTA"))
            continue

        wt_seq, designable = designable_for(scaffold_id, method)
        native, designs = bio.split_native(bio.parse_fasta(fasta))
        if native.sequence != wt_seq:
            raise SystemExit(
                f"{cell_id}: the FASTA's native record does not match the "
                "LayerSelector cache wild type. These are not the same scaffold.")

        q_wt = charge_lib.net_charge(wt_seq)
        target = int(cell["target_charge"])
        cell_rows = []
        for rec in designs:
            sample_index = int(rec.name.rsplit("_", 1)[1])
            did = bio.design_id(scaffold_id, method, target, sample_index)
            q_actual = charge_lib.net_charge(rec.sequence)
            q_err, hit = charge_lib.charge_error(q_actual, target)
            n_mut, muts = charge_lib.count_mutations(wt_seq, rec.sequence)

            row = {
                "design_id": did, "scaffold_id": scaffold_id,
                "pdb_id": scaffold["pdb_id"], "fold_class": scaffold["fold_class"],
                "n_residues": len(wt_seq), "q_wt": q_wt,
                "method": method, "weights": cell.get("weights", ""),
                "hbond_filter": hbond_filter_for(cell),
                "seed": cell.get("seed", ""), "sample_index": sample_index,
                "delta_q_density": cell["delta_q_density"], "target_charge": target,
                "q_actual": q_actual, "q_error_abs": q_err, "hit_exact": hit,
                "n_mutations": n_mut, "mutations": ";".join(muts),
                "mut_per_charge": charge_lib.mut_per_charge(n_mut, q_actual, q_wt),
                "frac_designable_mutated": met.frac_designable_mutated(
                    wt_seq, rec.sequence, designable),
                "seq_identity_to_wt_designable": met.seq_identity(
                    wt_seq, rec.sequence, designable),
                "seq_identity_to_wt_global": met.seq_identity(wt_seq, rec.sequence),
                "mpnn_score_masked": rec.score if rec.score is not None else "",
                "mpnn_score_global": (rec.global_score
                                      if rec.global_score is not None else ""),
                "final_temperature": rec.temperature if rec.temperature is not None else "",
                "n_retries": cell.get("n_retries", ""),
                "wall_seconds": cell.get("wall_seconds", ""),
                "status": "ok", "fail_reason": "",
            }

            thread = threading.get((cell_id, sample_index))
            if thread:
                row["thread_tier"] = thread["thread_tier"]
                for col in ("d_reu_per_res", "d_fa_atr", "d_fa_rep", "d_fa_sol",
                            "d_fa_elec", "d_hbond_sc", "d_hbond_bb_sc",
                            "d_hbond_strong", "d_hbond_weak"):
                    if col in thread:
                        row[col] = thread[col]

            pred = esmfold.get(did)
            if pred and pred.get("status") == "ok":
                for src, dst in (("plddt_mean", "plddt_mean"),
                                 ("plddt_min_designable", "plddt_min_designable"),
                                 ("d_plddt_vs_wt_pred", "d_plddt_vs_wt_pred"),
                                 ("ca_rmsd_to_wt_crystal", "ca_rmsd_to_wt_crystal"),
                                 ("tm_score_to_wt_crystal", "tm_score_to_wt_crystal")):
                    row[dst] = pred.get(src, "")

            a3 = af3.get(did)
            if a3:
                row["af3_plddt"] = a3.get("af3_plddt", "")
                row["af3_ca_rmsd"] = a3.get("af3_ca_rmsd", "")
                row["af3_msa_mode"] = a3.get("af3_msa_mode", "")

            cell_rows.append(bio.validate_row(row))

        design_rows.extend(cell_rows)

        # ---- Section 10.2 diversity, per cell -------------------------------
        seqs = [r.sequence for r in designs]
        div = met.cell_diversity(wt_seq, seqs, designable)
        threaded = [threading.get((cell_id, int(r["sample_index"])))
                    for r in cell_rows]
        threaded = [t for t in threaded if t]
        preds = [esmfold.get(r["design_id"]) for r in cell_rows]
        preds = [p for p in preds if p and p.get("status") == "ok"]

        def fnum(rows, col):
            out = []
            for r in rows:
                v = r.get(col, "")
                if v not in ("", None):
                    out.append(float(v))
            return out

        summary_rows.append({
            "scaffold_id": scaffold_id, "pdb_id": scaffold["pdb_id"],
            "fold_class": scaffold["fold_class"], "n_residues": len(wt_seq),
            "n_designable": len(designable), "q_wt": q_wt, "method": method,
            "weights": cell.get("weights", ""),
            "hbond_filter": hbond_filter_for(cell),
            "delta_q_density": cell["delta_q_density"], "target_charge": target,
            "n_designs": len(cell_rows),
            "n_hit_exact": sum(1 for r in cell_rows if r["hit_exact"]),
            "hit_rate": (sum(1 for r in cell_rows if r["hit_exact"]) / len(cell_rows)
                         if cell_rows else ""),
            "mean_q_error_abs": mean([r["q_error_abs"] for r in cell_rows]),
            "median_n_mutations": median([r["n_mutations"] for r in cell_rows]),
            "median_mut_per_charge": median([r["mut_per_charge"] for r in cell_rows]),
            "n_unique_mutation_sets": div["n_unique_mutation_sets"],
            "mean_pairwise_hamming": div["mean_pairwise_hamming"]
            if div["mean_pairwise_hamming"] is not None else "",
            "positional_entropy": div["positional_entropy"]
            if div["positional_entropy"] is not None else "",
            "designable_coverage": div["designable_coverage"]
            if div["designable_coverage"] is not None else "",
            "median_frac_designable_mutated": median(
                [r["frac_designable_mutated"] for r in cell_rows]),
            "median_seq_identity_to_wt_designable": median(
                [r["seq_identity_to_wt_designable"] for r in cell_rows]),
            "n_threaded": len(threaded),
            "median_d_reu_per_res": median([t.get("d_reu_per_res") for t in threaded]),
            "median_d_hbond_strong": median([t.get("d_hbond_strong") for t in threaded]),
            "median_d_hbond_weak": median([t.get("d_hbond_weak") for t in threaded]),
            "n_predicted": len(preds),
            "median_plddt_mean": median(fnum(preds, "plddt_mean")),
            "median_d_plddt_vs_wt_pred": median(fnum(preds, "d_plddt_vs_wt_pred")),
            "median_tm_score_to_wt_crystal": median(
                fnum(preds, "tm_score_to_wt_crystal")),
            "median_ca_rmsd_to_wt_crystal": median(
                fnum(preds, "ca_rmsd_to_wt_crystal")),
            "cell_wall_seconds": cell.get("wall_seconds", ""),
        })

    bio.write_csv(designs_csv, design_rows, bio.DESIGNS_COLUMNS)
    bio.write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    bio.write_csv(runtime_csv, runtime_rows, bio.RUNTIME_COLUMNS)

    log(f"wrote {designs_csv} with {len(design_rows)} designs")
    log(f"wrote {summary_csv} with {len(summary_rows)} cells")
    log(f"wrote {runtime_csv} with {len(runtime_rows)} jobs")
    n_thread = sum(1 for r in design_rows if r.get("thread_tier"))
    n_pred = sum(1 for r in design_rows if r.get("plddt_mean") not in ("", None))
    log(f"  threaded: {n_thread}/{len(design_rows)}   "
        f"predicted: {n_pred}/{len(design_rows)}")
    by_method: dict[str, int] = {}
    for r in design_rows:
        by_method[r["method"]] = by_method.get(r["method"], 0) + 1
    for m, n in sorted(by_method.items()):
        log(f"    {m:32s} {n}")
    if skipped:
        log(f"  {len(skipped)} cells contributed no rows:")
        for cell_id, why in skipped[:20]:
            log(f"    {cell_id}: {why}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(config_lib.load_benchmark(), args.dry_run)


if __name__ == "__main__":
    main()

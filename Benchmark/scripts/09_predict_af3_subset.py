#!/usr/bin/env python3
"""Phase 6b: AlphaFold3 on a stratified subset.

PLAN.md Section 9. Three stages:

  09_predict_af3_subset.py --select    # choose the subset, write af_input JSONs
  09_predict_af3_subset.py --submit    # one GPU job per JSON
  09_predict_af3_subset.py --collect   # results/af3.csv

**Single-sequence mode, and that is the methodological point.** Section 9 is
explicit: building an MSA for a heavily supercharged sequence retrieves wild-type
homologs and rescues the prediction with evolutionary signal the design does not
possess. Every JSON therefore carries `unpairedMsa: ""`, `pairedMsa: ""` and
`templates: []`, and the run passes `--norun_data_pipeline`. `af3_msa_mode` is
recorded as `single_sequence` in every output row, including the wild-type
controls, so the comparison is like for like.

**Two deviations from Section 9's subset arithmetic, both forced by earlier
decisions and neither silent.**

1. Section 9 budgets "5 scaffolds: eGFP plus one per fold class". That assumed
   four fold classes. Decision 11 dropped `loop_rich`, so the benchmark has three
   classes and this is eGFP plus three, which is four scaffolds. The subset is 4
   scaffolds x 4 methods x 2 targets plus 4 wild-type controls = 36 jobs, inside
   Section 9's stated 30 to 40 budget.
2. Section 9 names the method `random_charge`. The arm built in Phase 4 is
   `random_control`, the name used in `benchmark.yaml`, every sidecar and
   `designs.csv`. Same arm, one name, and the name in the frozen artifacts wins.

The per-class scaffold is the **median-difficulty** one by ESMFold wild-type
pLDDT, per Section 9, so the subset is not accidentally the easiest or hardest
members. The design folded in each cell is the **medoid** under the corrected
Section 11.3 procedure, selected in BLOSUM62 distance space rather than in an
embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402
from lib import selection as sel       # noqa: E402
from lib import structure as struct    # noqa: E402

AF3_METHODS = ["mpnn_soluble", "avnapsa", "rosetta", "random_control"]
AF3_DENSITIES = [-16, 16]
MSA_MODE = "single_sequence"

SUBSET_COLUMNS = ["job_name", "kind", "design_id", "scaffold_id", "method",
                  "delta_q_density", "target_charge", "sample_index", "sequence"]
AF3_COLUMNS = ["design_id", "job_name", "kind", "scaffold_id", "method",
               "delta_q_density", "target_charge", "af3_plddt", "af3_ptm",
               "af3_ca_rmsd", "af3_tm_score", "af3_msa_mode", "status",
               "fail_reason"]


def log(msg: str) -> None:
    print(msg, flush=True)


def root() -> Path:
    return config_lib.bench_path("predictions", "af3")


def subset_path() -> Path:
    return root() / "subset.csv"


# ---------------------------------------------------------------------------
# stage 1: choose the subset
# ---------------------------------------------------------------------------

def select_subset(cfg: dict, dry_run: bool) -> None:
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    esmfold_csv = config_lib.bench_path("results", "esmfold.csv")
    focus = cfg["selection"]["focus_scaffold"]["scaffold_id"]

    if dry_run:
        print(f"[dry-run] would read {manifest_path} and {esmfold_csv} "
              f"({'present' if esmfold_csv.exists() else 'NOT PRESENT, run 08 --collect first'})")
        print(f"[dry-run] would pick {focus} plus the median-WT-pLDDT scaffold in "
              "each of the 3 fold classes")
        print(f"[dry-run] would take methods {AF3_METHODS} at densities "
              f"{AF3_DENSITIES}, one medoid design per cell, plus a WT control "
              "per scaffold")
        print(f"[dry-run] would write {subset_path()} and one af_input JSON per job "
              f"with empty MSAs (msa_mode={MSA_MODE})")
        return

    if not esmfold_csv.exists():
        raise SystemExit(
            f"missing {esmfold_csv}. Section 9 picks the median-difficulty "
            "scaffold per fold class by ESMFold wild-type pLDDT, so Phase 6a must "
            "be collected first. Nothing here guesses at difficulty.")
    with open(manifest_path) as fh:
        manifest = {r["scaffold_id"]: r for r in csv.DictReader(fh)}
    with open(esmfold_csv) as fh:
        esm = [r for r in csv.DictReader(fh)]

    wt_plddt = {r["scaffold_id"]: float(r["plddt_mean"])
                for r in esm if r["kind"] == "wt" and r["status"] == "ok"
                and r["plddt_mean"] not in ("", None)}
    if not wt_plddt:
        raise SystemExit(f"{esmfold_csv} has no successful wild-type predictions.")

    chosen = [focus]
    for fold_class in sorted({r["fold_class"] for r in manifest.values()
                              if r["fold_class"]}):
        members = sorted(
            (s for s in manifest
             if manifest[s]["fold_class"] == fold_class and s in wt_plddt
             and s != focus),
            key=lambda s: (wt_plddt[s], s))
        if not members:
            log(f"  {fold_class}: no scaffold with a wild-type pLDDT, skipped")
            continue
        median_member = members[len(members) // 2]
        chosen.append(median_member)
        log(f"  {fold_class}: {len(members)} candidates, median-difficulty pick "
            f"{median_member} (WT pLDDT {wt_plddt[median_member]:.2f})")

    rows = []
    for scaffold_id in chosen:
        wt_seq = None
        for density in AF3_DENSITIES:
            for method in AF3_METHODS:
                cells = list(config_lib.bench_path("results", "cells").glob(
                    f"{scaffold_id}_{method}_q*.json"))
                match = None
                for path in cells:
                    with open(path) as fh:
                        cell = json.load(fh)
                    if (cell.get("status") == "ok"
                            and int(cell["delta_q_density"]) == density):
                        match = cell
                        break
                if match is None:
                    log(f"  no ok cell for {scaffold_id} {method} density {density:+d}")
                    continue
                fasta = config_lib.bench_path("designs", f"{match['cell_id']}.fa")
                native, designs = bio.split_native(bio.parse_fasta(fasta))
                wt_seq = native.sequence
                seqs = [d.sequence for d in designs]
                pick = sel.select(seqs)["medoid_index"]
                rec = designs[pick]
                sample_index = int(rec.name.rsplit("_", 1)[1])
                did = bio.design_id(scaffold_id, method,
                                    int(match["target_charge"]), sample_index)
                rows.append({
                    "job_name": did, "kind": "design", "design_id": did,
                    "scaffold_id": scaffold_id, "method": method,
                    "delta_q_density": density,
                    "target_charge": match["target_charge"],
                    "sample_index": sample_index, "sequence": rec.sequence,
                })
        if wt_seq is None:
            pkl = config_lib.bench_path("data", "parsed",
                                        f"{scaffold_id}_seq_indices.pkl")
            wt_seq = bio.load_designable_cache(pkl)[0]
        rows.append({
            "job_name": f"{scaffold_id}_WT", "kind": "wt",
            "design_id": f"{scaffold_id}_WT", "scaffold_id": scaffold_id,
            "method": "wt", "delta_q_density": "", "target_charge": "",
            "sample_index": "", "sequence": wt_seq,
        })

    input_dir = root() / "af_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = {
            "name": row["job_name"],
            "sequences": [{"protein": {
                "id": "A", "sequence": row["sequence"],
                # Section 9: no freshly built MSA. Empty strings plus no
                # templates is AF3's documented single-sequence configuration.
                "unpairedMsa": "", "pairedMsa": "", "templates": [],
            }}],
            "modelSeeds": [int(cfg["seeds"]["base_seed"]) % (2 ** 31)],
            "dialect": "alphafold3", "version": 1,
        }
        with open(input_dir / f"{row['job_name']}.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    bio.write_csv(subset_path(), rows, SUBSET_COLUMNS)
    n_design = sum(1 for r in rows if r["kind"] == "design")
    log(f"wrote {subset_path()}: {len(rows)} jobs "
        f"({n_design} designs, {len(rows) - n_design} wild-type controls)")
    log(f"wrote {len(rows)} JSONs to {input_dir}, msa_mode={MSA_MODE}")
    log(f"  scaffolds: {', '.join(chosen)}")


# ---------------------------------------------------------------------------
# stage 2: submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None) -> None:
    if not subset_path().exists():
        if dry_run:
            print(f"[dry-run] would need {subset_path()}; run --select first")
            return
        raise SystemExit(f"missing {subset_path()}. Run --select first.")
    with open(subset_path()) as fh:
        rows = list(csv.DictReader(fh))

    sbatch = config_lib.bench_path("slurm", "array_af3.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase6_af3")
    res = cluster["slurm"]["phase6_af3"]
    spec = array or f"0-{len(rows) - 1}%{cluster['slurm']['max_gpu_jobs']}"

    cmd = ["sbatch", f"--array={spec}",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT}",
           f"--partition={cluster['slurm'][res['partition_key']]}",
           f"--gres=gpu:{res['gpus']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           *config_lib.exclude_flag(cluster),
           f"--output={logdir}/%A_%a.out", f"--error={logdir}/%A_%a.out",
           str(sbatch)]
    if res.get("constraint"):
        cmd.insert(1, f"--constraint={res['constraint']}")

    if dry_run:
        print(f"[dry-run] would submit {len(rows)} AF3 jobs: {' '.join(cmd)}")
        return
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# stage 3: collect
# ---------------------------------------------------------------------------

def _resolve_job_dir(out_dir: Path, job_name: str) -> Path | None:
    """Find AF3's output directory for a job, whatever it renamed it to.

    AF3 sanitises the fold-input name into a directory name and does two things
    to it: it lowercases, and it DROPS the '+' sign. A job named
    `eGFP_mpnn_soluble_q+30_s08` lands in `egfp_mpnn_soluble_q30_s08`. Matching
    on `job_name.lower()` alone therefore found every negative-charge job and
    missed every positive one, which is exactly the 16 of 36 that came back with
    no model on the first collection.

    Candidates are tried in order of decreasing specificity, ending with a glob
    that tolerates any further character the sanitiser removes.
    """
    candidates = [job_name, job_name.lower(),
                  job_name.replace("+", ""), job_name.lower().replace("+", "")]
    for name in candidates:
        path = out_dir / name
        if path.is_dir():
            return path
    # Last resort: the sanitiser dropped something else. Match on the parts that
    # survive any of these transformations.
    stem = job_name.lower().replace("+", "").replace("-", "")
    for path in out_dir.iterdir():
        if path.is_dir() and path.name.lower().replace("+", "").replace("-", "") == stem:
            return path
    return None


def _find_model_cif(job_dir: Path) -> Path | None:
    """AF3 writes <name>_model.cif under its job directory."""
    candidates = sorted(job_dir.rglob("*_model.cif"))
    return candidates[0] if candidates else None


def collect(cfg: dict, dry_run: bool) -> None:
    out_csv = config_lib.bench_path("results", "af3.csv")
    out_dir = root() / "af_output"

    if dry_run:
        print(f"[dry-run] would read {subset_path()} and every {out_dir}/<job>/*_model.cif")
        print(f"[dry-run] would TM-align each onto its wild-type CRYSTAL chain and "
              f"write {out_csv}")
        return

    if not subset_path().exists():
        raise SystemExit(f"missing {subset_path()}. Run --select first.")
    with open(subset_path()) as fh:
        rows = list(csv.DictReader(fh))

    crystals: dict[str, tuple] = {}
    out_rows = []
    for row in rows:
        scaffold_id = row["scaffold_id"]
        rec = {c: "" for c in AF3_COLUMNS}
        rec.update({
            "design_id": row["design_id"], "job_name": row["job_name"],
            "kind": row["kind"], "scaffold_id": scaffold_id,
            "method": row["method"], "delta_q_density": row["delta_q_density"],
            "target_charge": row["target_charge"], "af3_msa_mode": MSA_MODE,
        })
        job_dir = _resolve_job_dir(out_dir, row["job_name"]) if out_dir.is_dir() else None
        cif = _find_model_cif(job_dir) if job_dir else None
        if cif is None:
            rec["status"] = "failed"
            rec["fail_reason"] = (f"no *_model.cif for {row['job_name']}"
                                  + (f" under {job_dir}" if job_dir else
                                     f"; no output directory under {out_dir}"))
            out_rows.append(rec)
            continue
        try:
            summary = sorted(job_dir.rglob("*summary_confidences.json"))
            if summary:
                with open(summary[0]) as fh:
                    conf = json.load(fh)
                if conf.get("ptm") is not None:
                    rec["af3_ptm"] = conf["ptm"]
            plddt = struct.read_plddt_per_residue(cif)
            rec["af3_plddt"] = sum(plddt) / len(plddt)
            if scaffold_id not in crystals:
                pdb = config_lib.bench_path("data", "scaffolds", scaffold_id,
                                            f"{scaffold_id}.pdb")
                crystals[scaffold_id] = struct.read_ca_trace(pdb)
            # The model is an mmCIF, not a PDB; read_ca_trace_any dispatches.
            coords, seq = struct.read_ca_trace_any(cif)
            ref_coords, ref_seq = crystals[scaffold_id]
            aln = struct.tm_align(coords, seq, ref_coords, ref_seq)
            rec["af3_ca_rmsd"] = aln["ca_rmsd"]
            rec["af3_tm_score"] = aln["tm_score"]
            rec["status"] = "ok"
        except Exception as exc:                                  # noqa: BLE001
            rec["status"] = "failed"
            rec["fail_reason"] = f"{type(exc).__name__}: {exc}"
        out_rows.append(rec)

    bio.write_csv(out_csv, out_rows, AF3_COLUMNS)
    n_ok = sum(1 for r in out_rows if r["status"] == "ok")
    log(f"wrote {out_csv} with {len(out_rows)} rows, {n_ok} ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select", action="store_true")
    group.add_argument("--submit", action="store_true")
    group.add_argument("--collect", action="store_true")
    parser.add_argument("--array", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.select:
        select_subset(cfg, args.dry_run)
    elif args.submit:
        submit(cluster, args.dry_run, args.array)
    elif args.collect:
        collect(cfg, args.dry_run)


if __name__ == "__main__":
    main()

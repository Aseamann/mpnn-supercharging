#!/usr/bin/env python3
"""Phase 6a: ESMFold2 prediction of every design, and the metrics off it.

PLAN.md Section 9. Three stages, run in order:

  08_predict_esmfold.py --emit-fasta   # one FASTA of every design plus every WT
  08_predict_esmfold.py --submit       # sharded GPU array
  08_predict_esmfold.py --collect      # results/esmfold.csv

Folding is done by `/projects/f_sdk94_1/Tools/ESMfold2/fold_fasta.py`, which is
read-only shared lab code. `slurm/array_esmfold.sbatch` is adapted from that
directory's `run_esmfold2.slurm` as Section 9 instructs, with the log and output
paths pointed into `Benchmark/` and `--also-pdb` added, because the per-residue
pLDDT and the TM-align superposition both need atom records this benchmark can
parse.

**Model choice.** The broad sweep uses `biohub/ESMFold2-Fast` at the tool's
default 3 loops / 50 sampling steps / 1 diffusion sample. The tool's README
advises the Fast model for screening large design sets and the full model for
anything going into a figure. This sweep is the screen: it covers every design
in the benchmark, and Section 9's own design puts the high-accuracy validation in
the AF3 subset instead. The model and its settings are recorded in every output
row, so the distinction stays visible rather than being an unstated default.

**Every record id is a `design_id`**, which is unique across the whole benchmark,
so the folded structures map back without a lookup table. `fold_fasta.py:88`
keeps `[A-Za-z0-9._+-]`, and a design id such as `eGFP_mpnn_soluble_q+2_s00`
contains nothing outside that set, so ids survive its sanitiser unchanged. That
is asserted at emit time rather than assumed.

Wild-type records are folded too, under the id `<scaffold_id>_WT`. Section 9
requires `d_plddt_vs_wt_pred`, the design's pLDDT minus the ESMFold pLDDT of the
wild-type sequence, which controls for scaffolds ESMFold simply predicts badly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402
from lib import structure as struct    # noqa: E402

FASTA_NAME = "designs.fasta"
MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = ["record_id", "kind", "scaffold_id", "method", "target_charge",
                    "delta_q_density", "sample_index", "n_residues"]

ESMFOLD_COLUMNS = [
    "record_id", "kind", "scaffold_id", "method", "target_charge",
    "delta_q_density", "sample_index", "esmfold_model", "esmfold_settings",
    "status", "plddt_mean", "plddt_min", "plddt_mean_designable",
    "plddt_min_designable", "d_plddt_vs_wt_pred", "ptm",
    "ca_rmsd_to_wt_crystal", "tm_score_to_wt_crystal",
    "n_residues", "fail_reason",
]

# Must match slurm/array_esmfold.sbatch. Recorded into every row.
ESMFOLD_MODEL = "biohub/ESMFold2-Fast"
ESMFOLD_SETTINGS = "loops=3 steps=50 diffusion_samples=1 seed=0"

SAFE_ID = re.compile(r"^[A-Za-z0-9._+-]+$")


def log(msg: str) -> None:
    print(msg, flush=True)


def say_would(msg: str) -> None:
    print(f"[dry-run] would {msg}", flush=True)


def out_root() -> Path:
    return config_lib.bench_path("predictions", "esmfold")


# ---------------------------------------------------------------------------
# stage 1: the FASTA
# ---------------------------------------------------------------------------

def emit_fasta(cfg: dict, dry_run: bool) -> None:
    fasta_path = out_root() / FASTA_NAME
    manifest_path = out_root() / MANIFEST_NAME

    if dry_run:
        say_would("read every results/cells/*.json with status ok and its designs/*.fa")
        say_would(f"write one record per design to {fasta_path}, id = design_id")
        say_would("add one record per scaffold for the wild-type sequence, "
                  "id = <scaffold_id>_WT, for d_plddt_vs_wt_pred")
        say_would(f"write {manifest_path} mapping every record id back to its cell")
        return

    records: list[tuple[str, str]] = []
    manifest: list[dict] = []
    wt_seen: dict[str, str] = {}

    for path in sorted(config_lib.bench_path("results", "cells").glob("*.json")):
        with open(path) as fh:
            cell = json.load(fh)
        if cell.get("status") != "ok":
            continue
        fasta = config_lib.bench_path("designs", f"{cell['cell_id']}.fa")
        if not fasta.exists():
            continue
        native, designs = bio.split_native(bio.parse_fasta(fasta))
        scaffold_id = cell["scaffold_id"]
        if scaffold_id not in wt_seen:
            wt_seen[scaffold_id] = native.sequence
        elif wt_seen[scaffold_id] != native.sequence:
            raise SystemExit(
                f"{scaffold_id}: two cells disagree about the wild-type sequence. "
                "Every arm must have been run against the same scaffold.")
        for rec in designs:
            sample_index = int(rec.name.rsplit("_", 1)[1])
            record_id = bio.design_id(scaffold_id, cell["method"],
                                      int(cell["target_charge"]), sample_index)
            records.append((record_id, rec.sequence))
            manifest.append({
                "record_id": record_id, "kind": "design",
                "scaffold_id": scaffold_id, "method": cell["method"],
                "target_charge": cell["target_charge"],
                "delta_q_density": cell["delta_q_density"],
                "sample_index": sample_index, "n_residues": len(rec.sequence),
            })

    for scaffold_id, seq in sorted(wt_seen.items()):
        record_id = f"{scaffold_id}_WT"
        records.append((record_id, seq))
        manifest.append({
            "record_id": record_id, "kind": "wt", "scaffold_id": scaffold_id,
            "method": "", "target_charge": "", "delta_q_density": "",
            "sample_index": "", "n_residues": len(seq),
        })

    unsafe = [r for r, _ in records if not SAFE_ID.match(r)]
    if unsafe:
        raise SystemExit(
            f"{len(unsafe)} record ids contain characters fold_fasta.py:88 would "
            f"rewrite, e.g. {unsafe[:3]}. Folded output could not be mapped back.")
    if len({r for r, _ in records}) != len(records):
        raise SystemExit("duplicate record ids; design_id is not unique.")

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as fh:
        for record_id, seq in records:
            fh.write(f">{record_id}\n{seq}\n")
    bio.write_csv(manifest_path, manifest, MANIFEST_COLUMNS)

    n_design = sum(1 for m in manifest if m["kind"] == "design")
    log(f"wrote {fasta_path} with {len(records)} records "
        f"({n_design} designs, {len(wt_seen)} wild types)")
    log(f"wrote {manifest_path}")


# ---------------------------------------------------------------------------
# stage 2: submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None, shards: int) -> None:
    fasta_path = out_root() / FASTA_NAME
    sbatch = config_lib.bench_path("slurm", "array_esmfold.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase6")
    res = cluster["slurm"]["phase6_esmfold"]
    throttle = cluster["slurm"]["max_gpu_jobs"]
    spec = array or f"0-{shards - 1}%{throttle}"

    # BM_NUM_SHARDS, not SLURM_ARRAY_TASK_COUNT: a partial rerun has a smaller
    # task count, and letting fold_fasta.py re-shard on it makes the reran tasks
    # select no records. See slurm/array_esmfold.sbatch.
    cmd = ["sbatch", f"--array={spec}",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT},BM_NUM_SHARDS={shards}",
           f"--partition={cluster['slurm'][res['partition_key']]}",
           f"--gres=gpu:{res['gpus']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           *config_lib.exclude_flag(cluster),
           f"--output={logdir}/%A_%a.out", f"--error={logdir}/%A_%a.out",
           str(sbatch)]

    if dry_run:
        say_would(f"submit {shards} shards of {fasta_path} "
                  f"({'present' if fasta_path.exists() else 'NOT PRESENT'})")
        say_would(f"run: {' '.join(cmd)}")
        say_would("note fold_fasta.py skips records whose .cif already exists, so a "
                  "resubmission resumes rather than refolding")
        return

    if not fasta_path.exists():
        raise SystemExit(f"missing {fasta_path}. Run --emit-fasta first.")
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# stage 3: collect
# ---------------------------------------------------------------------------

def collect(cfg: dict, dry_run: bool) -> None:
    manifest_path = out_root() / MANIFEST_NAME
    struct_dir = out_root() / "structures"
    out_csv = config_lib.bench_path("results", "esmfold.csv")

    if dry_run:
        say_would(f"read {manifest_path} and every {struct_dir}/summary.shard*.tsv")
        say_would(f"read per-residue pLDDT from {struct_dir}/<record_id>.pdb")
        say_would("TM-align every prediction onto its wild-type CRYSTAL chain in "
                  "data/scaffolds/, not onto the threaded model")
        say_would(f"write {out_csv} with columns {ESMFOLD_COLUMNS}")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Run --emit-fasta first.")
    with open(manifest_path) as fh:
        manifest = list(csv.DictReader(fh))

    summaries: dict[str, dict] = {}
    for shard in sorted(struct_dir.glob("summary.shard*.tsv")):
        with open(shard) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                summaries[row["id"]] = row
    if not summaries:
        raise SystemExit(f"no summary.shard*.tsv under {struct_dir}. Has the array run?")

    # Wild-type crystal CA traces and designable sets, one read per scaffold.
    crystals: dict[str, tuple] = {}
    designable: dict[str, list[int]] = {}
    for scaffold_id in sorted({m["scaffold_id"] for m in manifest}):
        pdb = config_lib.bench_path("data", "scaffolds", scaffold_id, f"{scaffold_id}.pdb")
        crystals[scaffold_id] = struct.read_ca_trace(pdb)
        cache = config_lib.bench_path("data", "parsed", f"{scaffold_id}_seq_indices.pkl")
        designable[scaffold_id] = bio.load_designable_cache(cache)[1]

    # Wild-type predicted pLDDT, needed before any design row can be written.
    #
    # Computed from the CIF through the same function the designs use, NOT from
    # the summary TSV's `plddt` column. The two are on different scales: the
    # summary reports 0.9232 where the CIF B-factor column reports 92.32.
    # Subtracting one from the other made d_plddt_vs_wt_pred average +73.55,
    # which is arithmetically impossible for a difference of two pLDDT values and
    # is how the mismatch was caught. Reading both through one function is what
    # keeps them on one scale.
    wt_plddt: dict[str, float] = {}
    for row in manifest:
        if row["kind"] != "wt":
            continue
        summary = summaries.get(row["record_id"])
        if not (summary and summary["status"] == "ok"):
            continue
        try:
            values = struct.read_plddt_per_residue(
                struct_dir / f"{row['record_id']}.cif")
            wt_plddt[row["scaffold_id"]] = sum(values) / len(values)
        except Exception as exc:                                  # noqa: BLE001
            log(f"  wild type {row['record_id']}: pLDDT unreadable ({exc}); "
                "its scaffold's designs get no d_plddt_vs_wt_pred")

    rows, n_missing_wt = [], 0
    for entry in manifest:
        record_id = entry["record_id"]
        scaffold_id = entry["scaffold_id"]
        out = {c: "" for c in ESMFOLD_COLUMNS}
        out.update({
            "record_id": record_id, "kind": entry["kind"],
            "scaffold_id": scaffold_id, "method": entry["method"],
            "target_charge": entry["target_charge"],
            "delta_q_density": entry["delta_q_density"],
            "sample_index": entry["sample_index"],
            "esmfold_model": ESMFOLD_MODEL, "esmfold_settings": ESMFOLD_SETTINGS,
            "n_residues": entry["n_residues"],
        })
        summary = summaries.get(record_id)
        if summary is None:
            out["status"] = "failed"
            out["fail_reason"] = "no summary row; record was never folded"
            rows.append(out)
            continue
        if summary["status"] != "ok":
            out["status"] = "failed"
            out["fail_reason"] = f"fold_fasta status={summary['status']}"
            rows.append(out)
            continue

        pdb = struct_dir / f"{record_id}.pdb"
        cif = struct_dir / f"{record_id}.cif"
        try:
            out["ptm"] = summary["ptm"]
            # pLDDT from the CIF: the --also-pdb conversion zeroes the B-factors.
            # See lib/structure.read_plddt_per_residue.
            plddt = struct.read_plddt_per_residue(cif)
            metrics = struct.plddt_metrics(plddt, designable[scaffold_id])
            out.update({k: v for k, v in metrics.items() if k in ESMFOLD_COLUMNS})
            if scaffold_id in wt_plddt:
                out["d_plddt_vs_wt_pred"] = metrics["plddt_mean"] - wt_plddt[scaffold_id]
            else:
                n_missing_wt += 1
            coords, seq = struct.read_ca_trace(pdb)
            ref_coords, ref_seq = crystals[scaffold_id]
            aln = struct.tm_align(coords, seq, ref_coords, ref_seq)
            out["ca_rmsd_to_wt_crystal"] = aln["ca_rmsd"]
            out["tm_score_to_wt_crystal"] = aln["tm_score"]
            out["status"] = "ok"
        except Exception as exc:                                  # noqa: BLE001
            out["status"] = "failed"
            out["fail_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(out)

    bio.write_csv(out_csv, rows, ESMFOLD_COLUMNS)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_design = sum(1 for r in rows if r["kind"] == "design")
    n_design_ok = sum(1 for r in rows if r["kind"] == "design" and r["status"] == "ok")
    log(f"wrote {out_csv} with {len(rows)} rows, {n_ok} ok")
    log(f"  designs: {n_design_ok}/{n_design} = "
        f"{100 * n_design_ok / n_design:.1f}% (Section 9 acceptance is >= 95%)")
    if n_missing_wt:
        log(f"  {n_missing_wt} rows have no d_plddt_vs_wt_pred: their scaffold's "
            "wild-type prediction did not succeed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-fasta", action="store_true")
    group.add_argument("--submit", action="store_true")
    group.add_argument("--collect", action="store_true")
    parser.add_argument("--shards", type=int, default=40,
                        help="Array size; fold_fasta.py splits the FASTA itself")
    parser.add_argument("--array", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_fasta:
        emit_fasta(cfg, args.dry_run)
    elif args.submit:
        submit(cluster, args.dry_run, args.array, args.shards)
    elif args.collect:
        collect(cfg, args.dry_run)


if __name__ == "__main__":
    main()

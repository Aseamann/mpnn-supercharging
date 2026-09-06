#!/usr/bin/env python3
"""Phase 5: thread designs onto their backbones, then score REU and hydrogen bonds.

PLAN.md Section 8. Threading is done by the repo's `threading_only.py`, which is
NOT modified; this driver builds its inputs, calls it, and scores the poses it
retains through `lib/rosetta_metrics.py`.

Tiering, per Section 8's budget table:

| Tier | Scope | num_relax |
|---|---|---|
| A | eGFP, all methods, all targets, all samples | 5 |
| B | the other 24 scaffolds, all methods, targets +/-8 and +/-16 density, 3 samples | 2 |
| C | everything else, sequence-level metrics only, not threaded |

**eGFP is not re-threaded in tier B.** Section 8 says tier B covers all
scaffolds, but eGFP is already covered at every target and every sample by tier A
at the deeper `num_relax`. Threading it again at `num_relax 2` would produce a
second pose for design ids that already have one, and `results/designs.csv` holds
one row per design with a single `thread_tier`. Tier A therefore takes
precedence and tier B spans the 24 non-focus scaffolds. Section 8 already
requires comparisons to be made within a tier, so nothing is lost, but a tier B
figure covers 24 scaffolds, not 25.

**Tier B takes sample indices 0, 1 and 2**, not a random three, so every method
contributes the same sample positions and the paired comparison stays paired.
AvNAPSA cells hold one sample and contribute that one.

Two kinds of task share the array. `wt_reference` tasks relax and score one
scaffold's wild type for one tier; `design` tasks thread and score one cell. The
references must exist first, so they are submitted and drained before the design
tasks. A design task with no matching reference fails rather than scoring
against something it invented.

Usage:
  07_thread_and_score.py --emit-tasks [--dry-run]
  07_thread_and_score.py --submit --kind wt [--dry-run]
  07_thread_and_score.py --submit --kind design [--array=...] [--dry-run]
  07_thread_and_score.py --time-one TASK_ID [--dry-run]
  07_thread_and_score.py --task 0 [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402
from lib import rosetta_metrics as rm  # noqa: E402

TASK_FILE = "slurm/tasks_phase5.tsv"
TASK_COLUMNS = ["task_id", "kind", "tier", "scaffold_id", "chain", "method",
                "delta_q_density", "target_charge", "cell_id", "n_samples",
                "num_relax", "seed"]

# Section 8's tier definitions, frozen here.
TIER_A_NUM_RELAX = 5
TIER_B_NUM_RELAX = 2
TIER_B_DENSITIES = {-16, -8, 8, 16}
TIER_B_SAMPLES = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def say_would(msg: str) -> None:
    print(f"[dry-run] would {msg}", flush=True)


def read_tasks() -> list[dict]:
    path = config_lib.bench_path(TASK_FILE)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run --emit-tasks first.")
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _task_for(task_index: int) -> dict:
    task = next((t for t in read_tasks() if int(t["task_id"]) == task_index), None)
    if task is None:
        raise SystemExit(f"no task with task_id={task_index} in {TASK_FILE}")
    return task


def _slurm_fields() -> dict:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }


def wt_reference_paths(tier: str, scaffold_id: str) -> tuple[Path, Path]:
    pdb = config_lib.bench_path("threaded", tier, "_wt", f"{scaffold_id}.pdb")
    rec = config_lib.bench_path("results", "threaded", tier, f"_wt_{scaffold_id}.json")
    return pdb, rec


# ---------------------------------------------------------------------------
# task file
# ---------------------------------------------------------------------------

def _completed_cells() -> list[dict]:
    """Every design cell that actually produced output, read from its sidecar.

    The task file is built from what Phases 2 to 4 wrote, not from the design
    matrix in benchmark.yaml, so a cell that failed upstream is never scheduled
    for threading and cannot silently contribute an empty row.
    """
    cells = []
    for path in sorted(config_lib.bench_path("results", "cells").glob("*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("status") != "ok":
            continue
        fasta = config_lib.bench_path("designs", f"{rec['cell_id']}.fa")
        if not fasta.exists():
            continue
        cells.append(rec)
    return cells


def emit_tasks(cfg: dict, dry_run: bool) -> None:
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    out = config_lib.bench_path(TASK_FILE)
    focus = cfg["selection"]["focus_scaffold"]["scaffold_id"]

    if dry_run:
        say_would(f"read every results/cells/*.json with status ok and its designs/*.fa")
        say_would(f"tier A: {focus} only, all methods, all targets, all samples, "
                  f"num_relax {TIER_A_NUM_RELAX}")
        say_would(f"tier B: every other scaffold, densities "
                  f"{sorted(TIER_B_DENSITIES)}, first {TIER_B_SAMPLES} samples, "
                  f"num_relax {TIER_B_NUM_RELAX}")
        say_would("add one wt_reference task per (scaffold, tier)")
        say_would(f"write {out} with columns {TASK_COLUMNS}")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Phase 1 must complete first.")
    with open(manifest_path) as fh:
        by_id = {r["scaffold_id"]: r for r in csv.DictReader(fh)}

    base_seed = cfg["seeds"]["base_seed"]
    cells = _completed_cells()
    if not cells:
        raise SystemExit("no completed design cells found in results/cells/")

    design_rows, scaffolds_in_tier = [], {"A": set(), "B": set()}
    for rec in cells:
        scaffold_id = rec["scaffold_id"]
        density = int(rec["delta_q_density"])
        if scaffold_id == focus:
            tier, num_relax, n_samples = "A", TIER_A_NUM_RELAX, int(rec["n_samples"])
        elif density in TIER_B_DENSITIES:
            tier, num_relax = "B", TIER_B_NUM_RELAX
            n_samples = min(TIER_B_SAMPLES, int(rec["n_samples"]))
        else:
            continue                      # tier C: sequence-level metrics only
        scaffolds_in_tier[tier].add(scaffold_id)
        design_rows.append({
            "kind": "design", "tier": tier,
            "scaffold_id": scaffold_id, "chain": by_id[scaffold_id]["chain"],
            "method": rec["method"], "delta_q_density": density,
            "target_charge": int(rec["target_charge"]), "cell_id": rec["cell_id"],
            "n_samples": n_samples, "num_relax": num_relax,
            "seed": bio.cell_seed(base_seed, scaffold_id,
                                  f"thread{tier}_{rec['method']}",
                                  int(rec["target_charge"])),
        })

    wt_rows = []
    for tier in ("A", "B"):
        for scaffold_id in sorted(scaffolds_in_tier[tier]):
            wt_rows.append({
                "kind": "wt_reference", "tier": tier,
                "scaffold_id": scaffold_id, "chain": by_id[scaffold_id]["chain"],
                "method": "", "delta_q_density": "", "target_charge": "",
                "cell_id": f"_wt_{tier}_{scaffold_id}", "n_samples": 1,
                "num_relax": TIER_A_NUM_RELAX if tier == "A" else TIER_B_NUM_RELAX,
                "seed": bio.cell_seed(base_seed, scaffold_id, f"wtref{tier}", 0),
            })

    # References first, so a single 0-based array index range covers them and the
    # design block stays contiguous underneath.
    rows = wt_rows + sorted(design_rows,
                            key=lambda r: (r["tier"], r["scaffold_id"],
                                           r["method"], r["target_charge"]))
    for i, row in enumerate(rows):
        row["task_id"] = i

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TASK_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    n_designs = sum(int(r["n_samples"]) for r in design_rows)
    log(f"wrote {out} with {len(rows)} tasks (array indices 0 to {len(rows) - 1})")
    log(f"  wt_reference: {len(wt_rows)} tasks, indices 0 to {len(wt_rows) - 1}")
    log(f"  design:       {len(design_rows)} tasks, indices {len(wt_rows)} to "
        f"{len(rows) - 1}, {n_designs} designs to thread")
    for tier in ("A", "B"):
        d = [r for r in design_rows if r["tier"] == tier]
        log(f"    tier {tier}: {len(d)} cells, {sum(int(r['n_samples']) for r in d)} "
            f"designs, {len(scaffolds_in_tier[tier])} scaffolds, "
            f"num_relax {TIER_A_NUM_RELAX if tier == 'A' else TIER_B_NUM_RELAX}")


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------

def run_wt_reference(cluster: dict, task: dict, dry_run: bool) -> None:
    scaffold_id = task["scaffold_id"]
    tier = task["tier"]
    num_relax = int(task["num_relax"])
    seed = int(task["seed"])
    scaffold_pdb = config_lib.bench_path("data", "scaffolds", scaffold_id,
                                         f"{scaffold_id}.pdb")
    out_pdb, out_rec = wt_reference_paths(tier, scaffold_id)

    if dry_run:
        say_would(f"read {scaffold_pdb} "
                  f"({'present' if scaffold_pdb.exists() else 'NOT PRESENT'})")
        say_would(f"relax it {num_relax} time(s) with an unrestricted movemap and "
                  "keep the lowest-energy pose (see lib/rosetta_metrics docstring: "
                  "threading_only.py does NOT relax a zero-mutation sequence)")
        say_would(f"write {out_pdb} and {out_rec}")
        return

    if not scaffold_pdb.exists():
        raise SystemExit(f"missing {scaffold_pdb}. Phase 1 must complete first.")

    record = {"kind": "wt_reference", "tier": tier, "scaffold_id": scaffold_id,
              "num_relax": num_relax, "seed": seed, **_slurm_fields()}
    started = time.time()
    try:
        record["pyrosetta_options"] = rm.init_pyrosetta(seed)
        record.update(rm.relax_wt_reference(scaffold_pdb, out_pdb, num_relax))
        record["pdb"] = str(out_pdb.relative_to(config_lib.BENCHMARK_ROOT))
        record["status"] = "ok"
        record["fail_reason"] = ""
    except Exception as exc:                                      # noqa: BLE001
        record["status"] = "failed"
        record["fail_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_seconds"] = round(time.time() - started, 2)

    out_rec.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rec, "w") as fh:
        json.dump(record, fh, indent=2)
    log(f"wt {tier}/{scaffold_id}: status={record['status']} "
        f"reu={record.get('reu_total')} crystal={record.get('reu_total_crystal')} "
        f"gain={record.get('reu_relaxation_gain')} {record['wall_seconds']}s")
    if record["status"] == "failed":
        log(f"  fail_reason: {record['fail_reason']}")


def _write_thread_input(cell_id: str, n_samples: int, dest: Path) -> list[str]:
    """Materialise the FASTA handed to threading_only.py.

    Only the design records, and only the first `n_samples` of them by
    sample_index. The wild type is not included: threading_only.py takes it from
    the input PDB, and passing it would only be skipped by --skip_wt.
    """
    fasta = config_lib.bench_path("designs", f"{cell_id}.fa")
    _, designs = bio.split_native(bio.parse_fasta(fasta))
    designs = sorted(designs, key=lambda d: int(d.name.rsplit("_", 1)[1]))
    chosen = designs[:n_samples]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        for rec in chosen:
            fh.write(f">{rec.name},charge={rec.charge}\n{rec.sequence}\n")
    return [rec.name for rec in chosen]


def run_design_cell(cluster: dict, task: dict, dry_run: bool) -> None:
    tier = task["tier"]
    cell_id = task["cell_id"]
    scaffold_id = task["scaffold_id"]
    num_relax = int(task["num_relax"])
    n_samples = int(task["n_samples"])
    seed = int(task["seed"])

    scaffold_pdb = config_lib.bench_path("data", "scaffolds", scaffold_id,
                                         f"{scaffold_id}.pdb")
    workdir = config_lib.bench_path("threaded", tier, cell_id)
    thread_fa = workdir / "input.fa"
    out_rec = config_lib.bench_path("results", "threaded", tier, f"{cell_id}.json")
    wt_pdb, wt_json = wt_reference_paths(tier, scaffold_id)
    script = Path(cluster["paths"]["threading_script"])
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    argv = [sys.executable, str(script),
            "-i", str(scaffold_pdb), "-f", str(thread_fa), "-o", str(workdir),
            "--num_relax", str(num_relax), "--workers", str(min(workers, n_samples)),
            "--skip_wt"]

    if dry_run:
        say_would(f"read the first {n_samples} design(s) from designs/{cell_id}.fa "
                  f"and write {thread_fa}")
        say_would(f"require the tier {tier} wild-type reference {wt_json} "
                  f"({'present' if wt_json.exists() else 'NOT PRESENT, run --kind wt first'})")
        say_would("run: " + " ".join(argv))
        say_would(f"score every retained pose with lib/rosetta_metrics and write {out_rec}")
        return

    if not scaffold_pdb.exists():
        raise SystemExit(f"missing {scaffold_pdb}.")
    if not wt_json.exists():
        raise SystemExit(
            f"missing wild-type reference {wt_json}. Submit --kind wt and let it "
            "drain before the design tasks; deltas are meaningless without it.")
    with open(wt_json) as fh:
        wt = json.load(fh)
    if wt.get("status") != "ok":
        raise SystemExit(f"{wt_json} has status={wt.get('status')}; cannot score against it.")

    record = {"kind": "design", "tier": tier, "cell_id": cell_id,
              "scaffold_id": scaffold_id, "method": task["method"],
              "target_charge": task["target_charge"],
              "delta_q_density": task["delta_q_density"],
              "num_relax": num_relax, "n_samples_requested": n_samples,
              "seed": seed, "wt_reference": str(wt_json.relative_to(config_lib.BENCHMARK_ROOT)),
              **_slurm_fields()}

    started = time.time()
    designs: list[dict] = []
    try:
        names = _write_thread_input(cell_id, n_samples, thread_fa)
        record["argv"] = argv
        proc = subprocess.run(argv, capture_output=True, text=True)
        record["returncode"] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError(f"threading_only.py exited {proc.returncode}: "
                               f"{proc.stderr.strip()[-500:]}")
        (workdir / "threading.log").write_text(proc.stdout)

        rm.init_pyrosetta(seed)
        import pyrosetta                                           # noqa: PLC0415
        sfxn = pyrosetta.get_fa_scorefxn()

        for name in names:
            pdb = workdir / f"{name}.pdb"
            if not pdb.exists():
                # --skip_wt drops any design identical to wild type. That is a
                # real outcome, recorded, not silently dropped from the count.
                designs.append({"design_name": name, "status": "not_threaded",
                                "reason": "no pose written; identical to wild type "
                                          "or skipped by threading_only.py"})
                continue
            scored = rm.score_pdb(pdb, sfxn)
            scored.update(rm.deltas_vs_wt(scored, wt))
            scored["design_name"] = name
            scored["sample_index"] = int(name.rsplit("_", 1)[1])
            scored["pdb"] = str(pdb.relative_to(config_lib.BENCHMARK_ROOT))
            scored["status"] = "ok"
            designs.append(scored)

        n_ok = sum(1 for d in designs if d["status"] == "ok")
        record["status"] = "ok" if n_ok else "failed"
        record["fail_reason"] = "" if n_ok else "no design produced a scored pose"
    except Exception as exc:                                      # noqa: BLE001
        record["status"] = "failed"
        record["fail_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_seconds"] = round(time.time() - started, 2)

    record["designs"] = designs
    record["n_scored"] = sum(1 for d in designs if d.get("status") == "ok")
    out_rec.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rec, "w") as fh:
        json.dump(record, fh, indent=2)

    log(f"{tier}/{cell_id}: status={record['status']} scored={record['n_scored']}"
        f"/{n_samples} {record['wall_seconds']}s -> {out_rec}")
    if record["status"] == "failed":
        log(f"  fail_reason: {record['fail_reason']}")


def run_task(cfg: dict, cluster: dict, task_index: int, dry_run: bool) -> None:
    task = _task_for(task_index)
    if task["kind"] == "wt_reference":
        run_wt_reference(cluster, task, dry_run)
    elif task["kind"] == "design":
        run_design_cell(cluster, task, dry_run)
    else:
        raise SystemExit(f"unknown kind {task['kind']!r} for task {task_index}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None, kind: str | None) -> None:
    task_path = config_lib.bench_path(TASK_FILE)
    tasks = read_tasks() if task_path.exists() else []
    sbatch = config_lib.bench_path("slurm", "array_thread.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase5")
    key = "phase5_wt" if kind == "wt" else "phase5_thread"
    res = cluster["slurm"][key]
    throttle = cluster["slurm"]["max_cpu_jobs"]

    if array:
        spec = array
    elif tasks and kind:
        want = "wt_reference" if kind == "wt" else "design"
        ids = sorted(int(t["task_id"]) for t in tasks if t["kind"] == want)
        if not ids:
            raise SystemExit(f"no tasks of kind {want!r} in {task_path}")
        spec = (f"{ids[0]}-{ids[-1]}%{throttle}"
                if ids == list(range(ids[0], ids[-1] + 1))
                else ",".join(str(i) for i in ids))
    elif tasks:
        spec = f"0-{len(tasks) - 1}%{throttle}"
    else:
        spec = f"0-<N-1>%{throttle}"

    cmd = ["sbatch", f"--array={spec}",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT}",
           f"--partition={cluster['slurm']['cpu_partition']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           *config_lib.exclude_flag(cluster),
           f"--output={logdir}/%A_%a.out", f"--error={logdir}/%A_%a.out",
           str(sbatch)]

    if dry_run:
        say_would(f"submit: {' '.join(cmd)}")
        if kind == "design":
            say_would("NOTE: every wt_reference task must have finished first. "
                      "A design task with no reference fails rather than "
                      "scoring against an unrelaxed crystal pose.")
        say_would(f"create {logdir} and log to %A_%a.out there")
        return

    if not tasks:
        raise SystemExit(f"missing {task_path}. Run --emit-tasks first.")
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tasks", action="store_true")
    group.add_argument("--task", type=int, help="Run one task by task_id")
    group.add_argument("--submit", action="store_true")
    group.add_argument("--time-one", type=int, metavar="TASK_ID",
                       help="Submit a single task for timing")
    parser.add_argument("--kind", choices=["wt", "design"], default=None,
                        help="Which block to submit")
    parser.add_argument("--array", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_tasks:
        emit_tasks(cfg, args.dry_run)
    elif args.task is not None:
        run_task(cfg, cluster, args.task, args.dry_run)
    elif args.submit:
        submit(cluster, args.dry_run, args.array, args.kind)
    elif args.time_one is not None:
        submit(cluster, args.dry_run, str(args.time_one), args.kind)


if __name__ == "__main__":
    main()

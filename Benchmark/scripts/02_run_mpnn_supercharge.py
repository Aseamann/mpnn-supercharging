#!/usr/bin/env python3
"""Phase 2: run the charge ladder through protein_mpnn_supercharge.py.

One array task is one cell of the design matrix: a (scaffold, arm, target
charge) triple, sampled 10 times in a single invocation.

Decision A, the seed shim. protein_mpnn_supercharge.py:642 draws its seed from
an unseeded RNG over 0..999, never prints it, and has no --seed flag, so there
is no way to fill the frozen `seed` column from the script's own output. main()
at line 600 is importable and guarded at line 923, so this driver seeds numpy
immediately before calling main(). That makes the internal draw deterministic
and knowable. Nothing under the repo root or /home/als515 is modified.

The invocation is the one frozen in PLAN.md Section 5, including -mhbond in the
primary arm. -mhbond REMOVES h-bond protection (argparse default False protects
them). That is the configuration the manuscript's results were produced under
and it is not to be "fixed". The hbond_filter column reads backwards from the
flag and is derived in exactly one place, lib/io.py.

Working directory matters. The script writes its LayerSelector cache to a
CWD-relative parsed/ directory (line 678) and the path is not configurable, so
each task chdirs into its arm's working directory. Not symlinked around.

Usage:
  02_run_mpnn_supercharge.py --emit-tasks
  02_run_mpnn_supercharge.py --task 0 --dry-run
  02_run_mpnn_supercharge.py --submit --dry-run
  02_run_mpnn_supercharge.py --time-one            # single-cell timing test
"""

from __future__ import annotations

import argparse
import contextlib
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

TASK_FILE = "slurm/tasks_phase2.tsv"
TASK_COLUMNS = ["task_id", "scaffold_id", "chain", "method", "weights",
                "weights_root", "mutate_hbonded_sidechains", "workdir",
                "delta_q_density", "delta_q", "target_charge", "n_samples", "seed"]


def log(msg: str) -> None:
    print(msg, flush=True)


def say_would(msg: str) -> None:
    print(f"[dry-run] would {msg}", flush=True)


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def read_tasks() -> list[dict]:
    path = config_lib.bench_path(TASK_FILE)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run --emit-tasks first.")
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


class _Tee:
    """Duplicate the script's stdout to a per-cell log.

    n_retries exists in no output file. It is the count of the
    "Increasing temperature to: " lines that line 462 prints, so the stdout of
    each cell has to survive somewhere the parser can find it. Relying on the
    Slurm log alone would tie the number to a file that gets rotated away.
    """

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.fh = open(path, "w")

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------------------
# task file
# ---------------------------------------------------------------------------

# Moved to lib/io.py on 2026-09-05 so Phase 2's mpnn_soluble_hbond_protected and
# Phase 3's rosetta_hbond_off resolve the identical scaffold set from one
# implementation. Kept as an alias so nothing that imported it from here breaks.
control_subset = bio.control_subset


def emit_tasks(cfg: dict, dry_run: bool) -> None:
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    out = config_lib.bench_path(TASK_FILE)
    resolved = cfg.get("resolved_targets")
    ladder = cfg["charge_ladder"]["delta_q_density"]
    mpnn_arms = [a for a in cfg["arms"] if a["name"].startswith("mpnn")]

    if dry_run:
        say_would(f"read {manifest_path} "
                  f"({'present' if manifest_path.exists() else 'NOT PRESENT'})")
        say_would("read the resolved_targets block from config/benchmark.yaml "
                  f"({'present' if resolved else 'NOT PRESENT, run 00 --stage targets'})")
        for arm in mpnn_arms:
            scope = "all scaffolds" if arm["scaffolds"] == "all" else (
                f"eGFP plus {arm.get('control_subset_per_class', 2)} per fold class")
            say_would(f"  arm {arm['name']}: {scope} x {len(ladder)} targets "
                      f"x {cfg['design']['n_samples']} samples, "
                      f"weights={arm.get('weights_root') or arm['weights']}, "
                      f"-mhbond {'passed' if arm['mutate_hbonded_sidechains'] else 'absent'}")
        say_would(f"write {out} with columns {TASK_COLUMNS}")
        say_would("derive one seed per cell via lib/io.cell_seed and record it in the "
                  "task file so a rerun of a single index reproduces its run")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Run 00_curate_scaffolds.py first.")
    if not resolved:
        raise SystemExit(
            "config/benchmark.yaml has no resolved_targets block. "
            "Run 00_curate_scaffolds.py --stage targets first."
        )

    manifest = read_csv(manifest_path)
    by_id = {r["scaffold_id"]: r for r in manifest}
    base_seed = cfg["seeds"]["base_seed"]
    rows, task_id = [], 0

    for arm in mpnn_arms:
        if arm["scaffolds"] == "all":
            ids = sorted(r["scaffold_id"] for r in manifest)
        else:
            ids = sorted(control_subset(manifest, arm.get("control_subset_per_class", 2)))
        for scaffold_id in ids:
            if scaffold_id not in resolved:
                raise SystemExit(
                    f"{scaffold_id} has no resolved targets. Rerun "
                    "00_curate_scaffolds.py --stage targets after changing the manifest."
                )
            for cell in resolved[scaffold_id]:
                rows.append({
                    "task_id": task_id,
                    "scaffold_id": scaffold_id,
                    "chain": by_id[scaffold_id]["chain"],
                    "method": arm["name"],
                    "weights": arm["weights"],
                    # Empty for the stock arms, which take the ProteinMPNN clone
                    # from cluster.yaml. Non-empty only for the biased-weight
                    # arms, which point at a shim directory under Benchmark/.
                    "weights_root": arm.get("weights_root", ""),
                    "mutate_hbonded_sidechains": arm["mutate_hbonded_sidechains"],
                    "workdir": arm["workdir"],
                    "delta_q_density": cell["delta_q_density"],
                    "delta_q": cell["delta_q"],
                    "target_charge": cell["target_charge"],
                    "n_samples": cfg["design"]["n_samples"],
                    "seed": bio.cell_seed(base_seed, scaffold_id, arm["name"],
                                          cell["target_charge"]),
                })
                task_id += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TASK_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    log(f"wrote {out} with {len(rows)} cells (array indices 0 to {len(rows) - 1})")
    for arm in mpnn_arms:
        n = sum(1 for r in rows if r["method"] == arm["name"])
        log(f"  {arm['name']}: {n} cells, {n * cfg['design']['n_samples']} designs")


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def weights_root(cluster: dict, task: dict) -> Path:
    """Directory passed as --path_to_weights for this cell.

    protein_mpnn_supercharge.py:618-627 uses this one path for two things: it
    joins {vanilla,soluble}_model_weights/<model>.pt onto it for the checkpoint,
    and it appends it to sys.path to import protein_mpnn_utils. So a directory
    holding symlinks to an alternative checkpoint and to the patched
    protein_mpnn_utils.py is enough to run a different weight set without
    touching the repo-root script, which has no --checkpoint flag and no else
    branch for an unrecognised --weights value.

    Empty weights_root, which is every arm written before 2026-09-05, keeps the
    previous behaviour of using the ProteinMPNN clone named in cluster.yaml.
    """
    root = str(task.get("weights_root", "") or "").strip()
    if not root:
        return Path(cluster["paths"]["proteinmpnn"])
    return config_lib.bench_path(*root.split("/"))


def weights_label(task: dict) -> str:
    """What goes in the `weights` column of designs.csv.

    The CLI value is "original" for the biased arms because the shim reuses the
    vanilla_model_weights directory name, which would make HyperMPNN and
    HaloMPNN indistinguishable from mpnn_vanilla_weights in the results. The
    recorded label is the shim directory name instead, so the column says which
    checkpoint actually produced the row.
    """
    root = str(task.get("weights_root", "") or "").strip()
    return root.rstrip("/").rsplit("/", 1)[-1] if root else task["weights"]


def build_argv(cfg: dict, cluster: dict, task: dict) -> list[str]:
    """The exact command line for one cell, per PLAN.md Section 5.

    Paths are absolute because the task chdirs into its arm working directory
    before calling main().
    """
    scaffold_id = task["scaffold_id"]
    argv = [
        "protein_mpnn_supercharge.py",
        "-i", str(config_lib.bench_path("data", "scaffolds", scaffold_id)),
        "-o", str(config_lib.bench_path("designs", f"{cell_name(task)}.fa")),
        "--chain_id", str(task["chain"]),
        "-c", str(task["target_charge"]),
        "-n", str(task["n_samples"]),
        "-t", str(cfg["design"]["temperature"]),
    ]
    if cfg["design"]["unrestrict"]:
        argv.append("-u")
    argv += [
        "--model", cfg["design"]["model"],
        "--weights", task["weights"],
        "--path_to_weights", str(weights_root(cluster, task)),
    ]
    if str(task["mutate_hbonded_sidechains"]).lower() == "true":
        argv.append("-mhbond")
    if cfg["design"]["mutate_glyprocys"]:
        argv.append("-gpc")
    if cfg["design"]["no_fastrelax"]:
        argv.append("-nofast")
    if cfg["charge_definition"]["add_histidine"]:
        argv.append("-addhis")
    argv.append("-v")            # required: n_retries is parsed from verbose stdout
    return argv


def cell_name(task: dict) -> str:
    return bio.cell_id(task["scaffold_id"], task["method"], int(task["target_charge"]))


def _format_argv(argv: list[str]) -> str:
    """Lay the command out one flag per line so a human can check it by eye."""
    lines, i = [], 0
    while i < len(argv):
        token = argv[i]
        takes_value = token.startswith("-") and i + 1 < len(argv) \
            and not argv[i + 1].startswith("-")
        # A negative charge argument looks like a flag; -c always takes its value.
        if token in ("-c", "-t", "-n", "-i", "-o", "-p") and i + 1 < len(argv):
            takes_value = True
        if takes_value:
            lines.append(f"{token} {argv[i + 1]}")
            i += 2
        else:
            lines.append(token)
            i += 1
    return " \\\n          ".join(lines)


def run_task(cfg: dict, cluster: dict, task_index: int, dry_run: bool) -> None:
    tasks = read_tasks()
    task = next((t for t in tasks if int(t["task_id"]) == task_index), None)
    if task is None:
        raise SystemExit(f"no task with task_id={task_index} in {TASK_FILE}")

    cell = cell_name(task)
    seed = int(task["seed"])
    workdir = config_lib.bench_path(task["workdir"])
    fasta = config_lib.bench_path("designs", f"{cell}.fa")
    cell_log = config_lib.bench_path("logs", "mpnn", f"{cell}.log")
    sidecar = config_lib.bench_path("results", "cells", f"{cell}.json")
    cache = workdir / "parsed" / f"{task['scaffold_id']}_seq_indices.pkl"
    argv = build_argv(cfg, cluster, task)

    if dry_run:
        say_would(f"cd {workdir}  (so the script's CWD-relative parsed/ cache lands there)")
        say_would(f"expect the Phase 1 cache at {cache} "
                  f"({'present' if cache.exists() else 'NOT PRESENT, Phase 1 must run first'})")
        say_would(f"np.random.seed({seed})   # Decision A shim")
        say_would("then call main(parse_args()) with:")
        print("      python " + _format_argv(argv), flush=True)
        say_would(f"tee stdout to {cell_log} and count "
                  f"{bio.RETRY_MARKER!r} lines from it for n_retries")
        say_would(f"write {fasta}")
        say_would(f"write {sidecar} with seed, wall_seconds, n_retries, "
                  f"final_temperature, hbond_filter="
                  f"{bio.hbond_filter_from_flag(str(task['mutate_hbonded_sidechains']).lower() == 'true')}, "
                  "status and the Slurm resources used")
        return

    if not cache.exists():
        raise SystemExit(
            f"missing LayerSelector cache {cache}.\n"
            "Phase 1 (01_prepare_structures.py) must complete for this arm's working "
            "directory first, or every task redoes the PyRosetta pass."
        )

    for path in (fasta, cell_log, sidecar):
        path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "cell_id": cell,
        "scaffold_id": task["scaffold_id"],
        "method": task["method"],
        "weights": weights_label(task),
        "hbond_filter": bio.hbond_filter_from_flag(
            str(task["mutate_hbonded_sidechains"]).lower() == "true"),
        "delta_q_density": int(task["delta_q_density"]),
        "target_charge": int(task["target_charge"]),
        "n_samples": int(task["n_samples"]),
        "seed": seed,
        "argv": argv,
        "workdir": str(workdir),
        "fasta": str(fasta),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "gpus": os.environ.get("SLURM_GPUS_ON_NODE", ""),
    }

    import numpy as np                                            # noqa: PLC0415
    sys.path.insert(0, str(cluster["paths"]["repo_root"]))
    import protein_mpnn_supercharge as sc                         # noqa: PLC0415

    original_cwd = Path.cwd()
    original_argv = sys.argv
    tee = _Tee(sys.stdout, cell_log)
    started = time.time()
    try:
        os.chdir(workdir)
        sys.argv = argv
        # The shim: seed numpy so the unseeded draw at line 642 is determined.
        np.random.seed(seed)
        with contextlib.redirect_stdout(tee):
            sc.main(sc.parse_args())
        record["status"] = "ok"
        record["fail_reason"] = ""
    except SystemExit as exc:
        record["status"] = "ok" if not exc.code else "failed"
        record["fail_reason"] = "" if not exc.code else f"SystemExit({exc.code})"
    except Exception as exc:                                      # noqa: BLE001
        record["status"] = "failed"
        record["fail_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_seconds"] = round(time.time() - started, 2)
        os.chdir(original_cwd)
        sys.argv = original_argv
        tee.flush()
        tee.close()

    # Retries and the FASTA are read back from disk. Nothing here is inferred.
    record["n_retries"] = bio.count_retries(cell_log) if cell_log.exists() else None
    record["temperature_ceiling_hit"] = (
        bio.temperature_exceeded(cell_log) if cell_log.exists() else None)

    if not fasta.exists():
        record["status"] = "failed"
        record["fail_reason"] = record.get("fail_reason") or "no FASTA written"
        record["n_designs"] = 0
        record["final_temperature"] = None
    else:
        try:
            native, designs = bio.split_native(bio.parse_fasta(fasta))
            record["n_designs"] = len(designs)
            record["q_wt_from_fasta"] = native.charge
            record["final_temperature"] = (
                max(d.temperature for d in designs) if designs else None)
            if not designs:
                record["status"] = "failed"
                record["fail_reason"] = "FASTA contains no sampled designs"
        except ValueError as exc:
            record["status"] = "failed"
            record["fail_reason"] = f"FASTA parse: {exc}"
            record["n_designs"] = 0
            record["final_temperature"] = None

    with open(sidecar, "w") as fh:
        json.dump(record, fh, indent=2)

    log(f"{cell}: status={record['status']} designs={record.get('n_designs')} "
        f"retries={record['n_retries']} final_T={record.get('final_temperature')} "
        f"{record['wall_seconds']}s -> {sidecar}")
    if record["status"] == "failed":
        log(f"  fail_reason: {record['fail_reason']}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None) -> None:
    task_path = config_lib.bench_path(TASK_FILE)
    # Under --dry-run the task file may not exist yet, because --emit-tasks was
    # itself a dry run. Report that rather than stopping.
    n_tasks = len(read_tasks()) if task_path.exists() else None
    throttle = cluster["slurm"]["max_cpu_jobs"]
    spec = array or (f"0-{n_tasks - 1}%{throttle}" if n_tasks else f"0-<N-1>%{throttle}")
    cmd, logdir = _sbatch_cmd(cluster, spec)

    if dry_run:
        count = f"{n_tasks}" if n_tasks else "an as yet unknown number of"
        say_would(f"submit {count} cells: {' '.join(cmd)}")
        if n_tasks is None:
            say_would(f"know N once {task_path} exists (run --emit-tasks for real)")
        say_would(f"throttle to {throttle} concurrent tasks, the CPU queue cap")
        say_would(f"create {logdir} and log to %A_%a.out there")
        say_would("NOTE: PLAN.md Section 5 requires timing one cell before sizing the "
                  "array. Run --time-one first and set slurm.phase2_mpnn.time from it.")
        return

    if n_tasks is None:
        raise SystemExit(f"missing {task_path}. Run --emit-tasks first.")
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _sbatch_cmd(cluster: dict, spec: str) -> tuple[list[str], Path]:
    sbatch = config_lib.bench_path("slurm", "array_mpnn.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase2")
    res = cluster["slurm"]["phase2_mpnn"]
    cmd = ["sbatch", f"--array={spec}",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT}", f"--partition={cluster['slurm']['cpu_partition']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           f"--output={logdir}/%A_%a.out", f"--error={logdir}/%A_%a.out",
           str(sbatch)]
    return cmd, logdir


def time_one(cluster: dict, dry_run: bool, task_index: int) -> None:
    """Submit a single cell so its walltime can set the array's --time."""
    cmd, logdir = _sbatch_cmd(cluster, str(task_index))
    if dry_run:
        say_would(f"submit exactly one cell for timing: {' '.join(cmd)}")
        say_would("then read wall_seconds from results/cells/<cell_id>.json and set "
                  "slurm.phase2_mpnn.time in cluster.yaml before submitting the array")
        return
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tasks", action="store_true")
    group.add_argument("--task", type=int, help="Run one cell by task_id")
    group.add_argument("--submit", action="store_true")
    group.add_argument("--time-one", type=int, nargs="?", const=0,
                       metavar="TASK_ID", help="Submit a single cell for timing")
    parser.add_argument("--array", default=None,
                        help="Override the --array spec, e.g. 17,43,91 to rerun failures")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_tasks:
        emit_tasks(cfg, args.dry_run)
    elif args.task is not None:
        run_task(cfg, cluster, args.task, args.dry_run)
    elif args.submit:
        submit(cluster, args.dry_run, args.array)
    elif args.time_one is not None:
        time_one(cluster, args.dry_run, args.time_one)


if __name__ == "__main__":
    main()

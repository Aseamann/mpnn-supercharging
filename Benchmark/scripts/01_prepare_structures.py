#!/usr/bin/env python3
"""Phase 1, second half: clean structures and build the LayerSelector cache.

Two things happen here, both per scaffold:

  1. The raw PDB is reduced to one chain with ligands and waters stripped, and
     written to data/scaffolds/<scaffold_id>/<scaffold_id>.pdb.
  2. The repo's LayerSelector pass runs once and its result is cached to
     <workdir>/parsed/<scaffold_id>_seq_indices.pkl, the exact file
     protein_mpnn_supercharge.py:685 later reads. n_designable is len(indices)
     from that pickle.

This is serial PyRosetta work that every later phase blocks on, so it runs as a
Slurm array rather than inline.

One cache per arm working directory, not one global cache. The repo script's
cache key does not encode mutate_hbonded_sidechains even though
parse_for_supercharge is called with it (protein_mpnn_supercharge.py:693), so a
shared cache would hand the h-bond-protected control arm the primary arm's
designable set and silently erase the thing that arm exists to measure.
See logs/ISSUES.md.

Usage:
  01_prepare_structures.py --emit-tasks            # write slurm/tasks_phase1.tsv
  01_prepare_structures.py --task 0                # run one array task
  01_prepare_structures.py --submit                # sbatch the array
  01_prepare_structures.py --collect               # fill n_designable into the manifest
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402

TASK_FILE = "slurm/tasks_phase1.tsv"
TASK_COLUMNS = ["task_id", "scaffold_id", "pdb_id", "chain", "workdir",
                "mutate_hbonded_sidechains"]

# PyRosetta options. -run:constant_seed is required by PLAN.md ground rule 3;
# the repo script's own init() does not pass it.
PYROSETTA_OPTIONS = "-run:constant_seed -constant_seed -jran 20260813 -mute all"


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


# ---------------------------------------------------------------------------
# task file
# ---------------------------------------------------------------------------

def control_subset(manifest: list[dict], per_class: int) -> set[str]:
    """eGFP plus `per_class` scaffolds from each fold class, for the control arm."""
    chosen = {r["scaffold_id"] for r in manifest if str(r["is_focus"]).lower() == "true"}
    by_class: dict[str, list[dict]] = {}
    for row in manifest:
        if row["fold_class"]:
            by_class.setdefault(row["fold_class"], []).append(row)
    for members in by_class.values():
        members.sort(key=lambda r: r["scaffold_id"])
        chosen.update(r["scaffold_id"] for r in members[:per_class])
    return chosen


def emit_tasks(cfg: dict, dry_run: bool) -> None:
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    out = config_lib.bench_path(TASK_FILE)

    if dry_run:
        say_would(f"read {manifest_path} "
                  f"({'present' if manifest_path.exists() else 'NOT PRESENT'})")
        say_would("emit one task per (scaffold, arm working directory); arms sharing a "
                  "working directory and h-bond setting share one cache and one task")
        for arm in cfg["arms"]:
            say_would(f"  arm {arm['name']}: workdir={arm['workdir']} "
                      f"mhbond={arm['mutate_hbonded_sidechains']} scaffolds={arm['scaffolds']}")
        say_would(f"write {out} with columns {TASK_COLUMNS}")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Run 00_curate_scaffolds.py first.")
    manifest = read_csv(manifest_path)

    # Distinct (workdir, mhbond) pairs; several arms can share one cache.
    seen: dict[tuple[str, bool], set[str]] = {}
    for arm in cfg["arms"]:
        key = (arm["workdir"], bool(arm["mutate_hbonded_sidechains"]))
        if arm["scaffolds"] == "all":
            ids = {r["scaffold_id"] for r in manifest}
        else:
            ids = control_subset(manifest, arm.get("control_subset_per_class", 2))
        seen.setdefault(key, set()).update(ids)

    by_id = {r["scaffold_id"]: r for r in manifest}
    rows, task_id = [], 0
    for (workdir, mhbond), ids in sorted(seen.items()):
        for scaffold_id in sorted(ids):
            row = by_id[scaffold_id]
            rows.append({
                "task_id": task_id, "scaffold_id": scaffold_id,
                "pdb_id": row["pdb_id"], "chain": row["chain"],
                "workdir": workdir, "mutate_hbonded_sidechains": mhbond,
            })
            task_id += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TASK_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {out} with {len(rows)} tasks "
        f"(array indices 0 to {len(rows) - 1})")


# ---------------------------------------------------------------------------
# structure cleaning
# ---------------------------------------------------------------------------

def clean_structure(src: Path, chain_id: str, dest: Path) -> tuple[int, list[dict]]:
    """Write a single-chain, ligand-free, water-free copy renumbered 1..N.

    Returns (residue count, original-to-new numbering map).

    The renumbering is not cosmetic, it is required for correctness.
    surface_selector at protein_mpnn_supercharge.py:84 returns **PDB** numbering
    via `pose.pdb_info().pose2pdb(res)`, but line 412 then uses those values to
    index `pose.sequence()`, which is **pose** numbering:

        indices = [i for i in indices if seq[i-1] not in ['G', 'P', 'C']]

    The two agree only when the PDB is numbered exactly 1..N. Otherwise the
    behaviour splits two ways, and only one of them is visible:

      - numbering that runs past the sequence length raises IndexError and the
        task dies loudly (20 of the 25 scaffolds in this manifest),
      - numbering that is merely shifted or has gaps stays in range and silently
        applies the Gly/Pro/Cys protection to the WRONG residues, so protected
        positions become mutable and mutable ones become protected
        (2 of 25, e.g. 4lws_B numbered -2..85 and 2gux_A numbered -4..121).

    The second case is the dangerous one: no error, wrong designable set, and
    every downstream metric quietly computed against it.

    Renumbering the cleaned scaffold to 1..N makes PDB numbering and pose
    numbering identical, which neutralises the mismatch without touching the
    repo script, as required. eGFP is already numbered 1..231, so the
    manuscript's published results are not affected by this bug.

    The original numbering is not discarded: it is returned and written next to
    the structure so crystal-numbering cross-references stay possible.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: establish residue order and the renumbering map.
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(src) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  ") or line[21] != chain_id:
                continue
            if line[16] not in (" ", "A"):
                continue
            key = (line[22:26].strip(), line[26])       # (resseq, insertion code)
            if key not in seen:
                seen.add(key)
                order.append(key)
    if not order:
        raise SystemExit(f"{src}: no ATOM records for chain {chain_id!r}")

    renumber = {key: i for i, key in enumerate(order, start=1)}

    # Pass 2: rewrite with sequential numbering and blank insertion codes.
    kept = []
    with open(src) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  ") or line[21] != chain_id:
                continue
            if line[16] not in (" ", "A"):
                continue
            key = (line[22:26].strip(), line[26])
            kept.append(f"{line[:22]}{renumber[key]:>4d} {line[27:]}")

    with open(dest, "w") as fh:
        fh.writelines(kept)
        fh.write("TER\nEND\n")

    mapping = [{"new_resnum": new, "orig_resnum": key[0],
                "orig_icode": key[1].strip(), "chain": chain_id}
               for key, new in renumber.items()]
    return len(order), mapping


# ---------------------------------------------------------------------------
# one array task
# ---------------------------------------------------------------------------

def run_task(cfg: dict, cluster: dict, task_index: int, dry_run: bool) -> None:
    tasks = read_tasks()
    task = next((t for t in tasks if int(t["task_id"]) == task_index), None)
    if task is None:
        raise SystemExit(f"no task with task_id={task_index} in {TASK_FILE}")

    scaffold_id = task["scaffold_id"]
    mhbond = task["mutate_hbonded_sidechains"].lower() == "true"
    workdir = config_lib.bench_path(task["workdir"])
    scaffold_pdb = config_lib.bench_path("data", "scaffolds", scaffold_id, f"{scaffold_id}.pdb")
    cache = workdir / "parsed" / f"{scaffold_id}_seq_indices.pkl"

    if str(scaffold_id) == cfg["selection"]["focus_scaffold"]["scaffold_id"]:
        raw = Path(cluster["paths"]["egfp_pdb"])
    else:
        raw = config_lib.bench_path("data", "raw", f"{task['pdb_id']}.pdb")

    if dry_run:
        say_would(f"clean chain {task['chain']} of {raw} into {scaffold_pdb}, "
                  "renumbered 1..N so PDB numbering matches pose numbering "
                  "(see clean_structure docstring)")
        say_would(f"write the original numbering map to "
                  f"{scaffold_pdb.parent}/{scaffold_id}_numbering.csv")
        say_would(f"init PyRosetta with {PYROSETTA_OPTIONS!r}")
        say_would(f"call parse_for_supercharge(mutate_glyprocys="
                  f"{cfg['design']['mutate_glyprocys']}, mutate_strong_hbond={mhbond}, "
                  f"no_fastrelax={cfg['design']['no_fastrelax']}, add_histidine="
                  f"{cfg['charge_definition']['add_histidine']})")
        say_would(f"write the cache to {cache}")
        say_would(f"record n_designable = len(indices) into {workdir / 'parsed'}"
                  f"/{scaffold_id}_designable.json")
        return

    started = time.time()
    n_res, numbering = clean_structure(raw, task["chain"], scaffold_pdb)
    # Keep the crystal numbering recoverable; see clean_structure's docstring.
    bio.write_csv(scaffold_pdb.parent / f"{scaffold_id}_numbering.csv", numbering,
                  ["new_resnum", "orig_resnum", "orig_icode", "chain"])
    orig_first = numbering[0]["orig_resnum"] if numbering else "?"
    orig_last = numbering[-1]["orig_resnum"] if numbering else "?"
    log(f"cleaned {raw} chain {task['chain']} -> {scaffold_pdb} "
        f"({n_res} residues, renumbered {orig_first}..{orig_last} -> 1..{n_res})")

    # Initialise PyRosetta ourselves so -run:constant_seed is set, then mark the
    # repo module as already initialised so its own init(silent=True) is skipped.
    import pyrosetta                                              # noqa: PLC0415
    pyrosetta.init(options=PYROSETTA_OPTIONS, silent=True)

    sys.path.insert(0, str(cluster["paths"]["repo_root"]))
    import protein_mpnn_supercharge as sc                         # noqa: PLC0415
    sc._pyrosetta_initialised = True

    wt_seq, indices, net_charge_init = sc.parse_for_supercharge(
        str(scaffold_pdb),
        None,                                     # catalytic
        None,                                     # distance
        cfg["design"]["mutate_glyprocys"],
        mhbond,
        cfg["design"]["no_fastrelax"],
        cfg["charge_definition"]["add_histidine"],
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump((wt_seq, indices, net_charge_init), fh)
    # The repo script writes this alongside the pickle; match it so a later run
    # of the script finds the cache in the state it expects.
    with open(cache.with_suffix("").with_suffix(".txt"), "w") as fh:
        fh.write(f"{wt_seq}\n{indices}\n")

    import json                                                   # noqa: PLC0415
    summary = {
        "scaffold_id": scaffold_id,
        "workdir": task["workdir"],
        "mutate_hbonded_sidechains": mhbond,
        "n_residues": len(wt_seq),
        "n_designable": len(indices),
        "q_wt_from_pkl": net_charge_init,
        "wall_seconds": round(time.time() - started, 2),
        "pyrosetta_options": PYROSETTA_OPTIONS,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }
    with open(cache.parent / f"{scaffold_id}_designable.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    log(f"{scaffold_id}: n_residues={len(wt_seq)} n_designable={len(indices)} "
        f"q_wt={net_charge_init} in {summary['wall_seconds']}s -> {cache}")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def collect(cfg: dict, dry_run: bool) -> None:
    """Fill n_designable into the manifest from the primary arm's cache."""
    import json                                                   # noqa: PLC0415

    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    primary = config_lib.primary_arm(cfg)
    parsed_dir = config_lib.bench_path(primary["workdir"], "parsed")

    if dry_run:
        say_would(f"read n_designable from {parsed_dir}/<scaffold>_designable.json "
                  f"(primary arm {primary['name']})")
        say_would(f"fill the n_designable column of {manifest_path} and leave any "
                  "scaffold whose cache is missing empty rather than guessing")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")
    rows = read_csv(manifest_path)

    filled, missing = 0, []
    for row in rows:
        summary_path = parsed_dir / f"{row['scaffold_id']}_designable.json"
        if not summary_path.exists():
            missing.append(row["scaffold_id"])
            continue
        with open(summary_path) as fh:
            summary = json.load(fh)
        row["n_designable"] = summary["n_designable"]
        # q_wt in the manifest came from the sequence in the filter stage; the
        # pickle carries the script's own value. A disagreement means the two
        # are reading different residues and must not be papered over.
        if int(row["q_wt"]) != int(summary["q_wt_from_pkl"]):
            log(f"  WARNING {row['scaffold_id']}: manifest q_wt={row['q_wt']} but "
                f"LayerSelector pass reports {summary['q_wt_from_pkl']}")
        filled += 1

    bio.write_csv(manifest_path, rows, bio.MANIFEST_COLUMNS)
    log(f"filled n_designable for {filled}/{len(rows)} scaffolds")
    if missing:
        log(f"still missing ({len(missing)}): {', '.join(missing[:10])}"
            f"{' ...' if len(missing) > 10 else ''}")
        log("rerun those array indices; n_designable stays empty until computed")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None) -> None:
    task_path = config_lib.bench_path(TASK_FILE)
    # Under --dry-run the task file may not exist yet, because --emit-tasks was
    # itself a dry run. Report that rather than stopping.
    n_tasks = len(read_tasks()) if task_path.exists() else None
    sbatch = config_lib.bench_path("slurm", "array_layerselect.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase1")
    res = cluster["slurm"]["phase1_layerselect"]
    throttle = cluster["slurm"]["max_cpu_jobs"]
    spec = array or (f"0-{n_tasks - 1}%{throttle}" if n_tasks else f"0-<N-1>%{throttle}")
    # %A_%a keeps the mapping from array task back to cell unambiguous.
    cmd = ["sbatch", f"--array={spec}",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT}", f"--partition={cluster['slurm']['cpu_partition']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           f"--output={logdir}/%A_%a.out", f"--error={logdir}/%A_%a.out",
           str(sbatch)]

    if dry_run:
        count = f"{n_tasks}" if n_tasks else "an as yet unknown number of"
        say_would(f"submit {count} tasks: {' '.join(cmd)}")
        if n_tasks is None:
            say_would(f"know N once {task_path} exists (run --emit-tasks for real)")
        say_would(f"create {logdir} and log to %A_%a.out there")
        return

    if n_tasks is None:
        raise SystemExit(f"missing {task_path}. Run --emit-tasks first.")
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tasks", action="store_true")
    group.add_argument("--task", type=int, help="Run one array task by task_id")
    group.add_argument("--submit", action="store_true")
    group.add_argument("--collect", action="store_true")
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
    elif args.collect:
        collect(cfg, args.dry_run)


if __name__ == "__main__":
    main()

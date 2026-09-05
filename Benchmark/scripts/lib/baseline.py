"""Shared runner for the Phase 3 classical baselines.

Both baselines drive the same PyRosetta mover,
`pyrosetta.rosetta.protocols.design_opt.Supercharge`, reached through
`Former_Methods/eGFP/run_pyrosetta_supercharge.py`. That script is the reference
for every mover setting used here and is NOT modified.

Decision B is resolved by this module existing: PyRosetta does expose the
supercharge protocol, with all 22 methods the reference script and PLAN.md
Section 6 rely on, so the baseline does not have to be cut.

Two deviations from a literal reading, both approved 2026-08-13:

1. **Direction comes from `delta_q`, not from the sign of the target.**
   `run_pyrosetta_supercharge.py:76` and `:91` select positive or negative
   supercharging with `if charge >= 0`, where `charge` is the absolute target
   net charge. That is only equivalent to "raise or lower the charge" when the
   scaffold's WT charge sits on the other side of zero from the target. Across
   this benchmark's grid it contradicts the required direction in 14 of 200
   cells per arm, for example 1a1x_A with q_wt -7 and target -3 needs the
   charge RAISED by 4, while the sign test picks negative supercharging and can
   only lower it, so the target is unreachable by construction. Those 14 cells
   would have failed for harness reasons and read as the baseline
   underperforming. Direction is therefore taken from `sign(delta_q)`.
   Every other mover setting is the reference script's, value for value.

2. **`surface_residue_cutoff(16)` on the Rosetta arm.** PLAN.md Section 6 names
   `-surface_atom_cutoff 120`, but in the Supercharge mover that parameter
   governs the AvNAPSA sequence-based surface definition, while
   `surface_residue_cutoff` (a neighbour count) governs the score-based Rosetta
   mode. PLAN.md conflated the two. The reference script uses the mode-correct
   one and both are Rosetta defaults.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from lib import charge as charge_lib
from lib import config as config_lib
from lib import io as bio

TASK_FILE = "slurm/tasks_phase3.tsv"
TASK_COLUMNS = ["task_id", "scaffold_id", "chain", "method", "delta_q_density",
                "delta_q", "target_charge", "nstruct", "seed"]

# PyRosetta options. PLAN.md ground rule 3 requires a recorded, constant seed;
# the reference script's bare `pyrosetta.init()` does not set one.
PYROSETTA_BASE_OPTIONS = "-run:constant_seed -constant_seed -mute all"

# -run:jran is parsed as a SIGNED 32-bit integer. lib/io.cell_seed returns up to
# 2**32-2, so roughly half of all cell seeds overflow it and PyRosetta exits with
# "Illegal value for integer option -run:jran". Rather than narrowing cell_seed,
# which would silently change every seed already recorded in the completed
# Phase 2 runs, the cell seed stays canonical and this is the value actually
# handed to PyRosetta. Both are written to the sidecar.
JRAN_MAX = 2 ** 31 - 1


def jran_for(seed: int) -> int:
    """Fold a cell seed into the signed 32-bit range -run:jran accepts."""
    return seed % JRAN_MAX


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


def load_reference_script(path: str | Path):
    """Import Former_Methods/eGFP/run_pyrosetta_supercharge.py without running it.

    The module is guarded by `if __name__ == '__main__'`, so importing is safe.
    Nothing in it is modified; it is loaded so the mover settings below can be
    checked against the real thing rather than a copy that could drift.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"reference script not found at {path}")
    spec = importlib.util.spec_from_file_location("run_pyrosetta_supercharge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_subset_per_class(cfg: dict) -> int:
    """How many scaffolds per fold class the control subset takes.

    Read off the MPNN control arm rather than duplicated here, so
    `rosetta_hbond_off` lands on exactly the same scaffolds as
    `mpnn_soluble_hbond_protected`. The h-bond on/off comparison in F9 is only
    readable if all four series sit on one scaffold set.
    """
    for arm in cfg["arms"]:
        if arm.get("scaffolds") == "control_subset":
            return int(arm.get("control_subset_per_class", 2))
    raise SystemExit(
        "no arm in config/benchmark.yaml uses scaffolds: control_subset, so the "
        "baseline control subset cannot be tied to the MPNN one.")


def configure_mover(mover, target_charge: int, delta_q: int, avnapsa: bool,
                    hbond_protect: bool = True) -> dict:
    """Apply the reference script's settings, with direction from delta_q.

    Mirrors `super_charge_pdb` at run_pyrosetta_supercharge.py:65-105 value for
    value. The only change is the direction test: the reference uses
    `charge >= 0`, this uses `delta_q > 0`. See this module's docstring.

    `hbond_protect` is the score-based mode's `dont_mutate_hbonded_sidechains`.
    It defaults to True, which is the reference script's value and what the
    `rosetta` arm has always run, so nothing already recorded changes. Setting
    it False is the direct analogue of passing `-mhbond` on the MPNN side: it
    lets h-bonded sidechains mutate. That is what the `rosetta_hbond_off` arm
    exists to measure, because the primary MPNN arm passes `-mhbond` and the
    `rosetta` arm does not, so F9 was comparing two arms that differ on more
    than the design method. It has no effect in AvNAPSA mode, which uses a
    sequence-based surface definition and never consults this switch.

    Returns the settings actually applied, for the sidecar record.
    """
    if delta_q == 0:
        raise ValueError("delta_q == 0 has no supercharging direction")
    raise_charge = delta_q > 0

    mover.target_net_charge(target_charge)
    mover.target_net_charge_active(True)
    applied = {"target_net_charge": target_charge, "target_net_charge_active": True,
               "direction": "positive" if raise_charge else "negative",
               "direction_from": "delta_q"}

    if avnapsa:
        # AvNAPSA mode: deterministic, sequence-based surface definition.
        # surface_atom_cutoff keeps its Rosetta default of 120 here, which is
        # the parameter PLAN.md Section 6 names.
        if raise_charge:
            mover.AvNAPSA_positive(True)
            applied["AvNAPSA_positive"] = True
        else:
            mover.AvNAPSA_negative(True)
            applied["AvNAPSA_negative"] = True
        return applied

    # Rosetta score-based mode, reference script lines 82-102.
    mover.surface_residue_cutoff(16)
    mover.pre_packminpack(True)
    mover.dont_mutate_glyprocys(True)
    mover.dont_mutate_correct_charge(True)
    mover.dont_mutate_hbonded_sidechains(hbond_protect)
    mover.compare_residue_energies_all(False)
    mover.compare_residue_energies_mut(True)
    applied.update({"surface_residue_cutoff": 16, "pre_packminpack": True,
                    "dont_mutate_glyprocys": True, "dont_mutate_correct_charge": True,
                    "dont_mutate_hbonded_sidechains": hbond_protect,
                    "compare_residue_energies_all": False,
                    "compare_residue_energies_mut": True})

    if raise_charge:
        mover.include_arg(True)
        mover.include_lys(True)
        mover.refweight_arg(-1.98)
        mover.refweight_lys(-1.65)
        applied.update({"include_arg": True, "include_lys": True,
                        "refweight_arg": -1.98, "refweight_lys": -1.65})
    else:
        mover.include_asp(True)
        mover.include_glu(True)
        mover.refweight_asp(-0.6)
        mover.refweight_glu(-0.8)
        applied.update({"include_asp": True, "include_glu": True,
                        "refweight_asp": -0.6, "refweight_glu": -0.8})
    return applied


# ---------------------------------------------------------------------------
# task file
# ---------------------------------------------------------------------------

def emit_tasks(cfg: dict, dry_run: bool) -> None:
    """One task per (method, scaffold, target). Both baselines share one file."""
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    out = config_lib.bench_path(TASK_FILE)
    resolved = cfg.get("resolved_targets")
    arms = cfg["baseline_arms"]

    if dry_run:
        say_would(f"read {manifest_path} "
                  f"({'present' if manifest_path.exists() else 'NOT PRESENT'})")
        say_would("read resolved_targets from config/benchmark.yaml "
                  f"({'present' if resolved else 'NOT PRESENT'})")
        for arm in arms:
            scope = ("all scaffolds" if arm.get("scaffolds", "all") == "all"
                     else "eGFP plus the fold-class control subset")
            say_would(f"  arm {arm['name']}: {scope} x "
                      f"{len(cfg['charge_ladder']['delta_q_density'])} targets, "
                      f"nstruct={arm['nstruct']}, avnapsa={arm['avnapsa']}, "
                      f"hbond_protect={arm.get('hbond_protect', True)}")
        say_would(f"write {out} with columns {TASK_COLUMNS}")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Phase 1 must complete first.")
    if not resolved:
        raise SystemExit("config/benchmark.yaml has no resolved_targets block.")

    manifest = read_csv(manifest_path)
    by_id = {r["scaffold_id"]: r for r in manifest}
    base_seed = cfg["seeds"]["base_seed"]
    rows, task_id = [], 0

    subset = bio.control_subset(manifest, control_subset_per_class(cfg))

    for arm in arms:
        ids = (sorted(by_id) if arm.get("scaffolds", "all") == "all"
               else sorted(subset))
        for scaffold_id in ids:
            for cell in resolved[scaffold_id]:
                rows.append({
                    "task_id": task_id,
                    "scaffold_id": scaffold_id,
                    "chain": by_id[scaffold_id]["chain"],
                    "method": arm["name"],
                    "delta_q_density": cell["delta_q_density"],
                    "delta_q": cell["delta_q"],
                    "target_charge": cell["target_charge"],
                    "nstruct": arm["nstruct"],
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
    for arm in arms:
        n = sum(1 for r in rows if r["method"] == arm["name"])
        log(f"  {arm['name']}: {n} cells, nstruct {arm['nstruct']}, "
            f"{n * arm['nstruct']} structures, "
            f"hbond_protect={arm.get('hbond_protect', True)}")


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def run_task(cfg: dict, cluster: dict, task_index: int, method: str,
             dry_run: bool) -> None:
    tasks = read_tasks()
    task = next((t for t in tasks
                 if int(t["task_id"]) == task_index and t["method"] == method), None)
    if task is None:
        raise SystemExit(
            f"no task with task_id={task_index} and method={method!r} in {TASK_FILE}")

    arm = next(a for a in cfg["baseline_arms"] if a["name"] == method)
    avnapsa = bool(arm["avnapsa"])
    hbond_protect = bool(arm.get("hbond_protect", True))
    scaffold_id = task["scaffold_id"]
    target = int(task["target_charge"])
    delta_q = int(task["delta_q"])
    nstruct = int(task["nstruct"])
    seed = int(task["seed"])

    cell = bio.cell_id(scaffold_id, method, target)
    scaffold_pdb = config_lib.bench_path("data", "scaffolds", scaffold_id,
                                         f"{scaffold_id}.pdb")
    # The mover writes fixed-name files (resfile_output_Rsc, resfile_output_Asc)
    # and its own PDBs into the CWD, so every cell needs its own directory or
    # concurrent array tasks overwrite each other.
    workdir = config_lib.bench_path("baselines", method, cell)
    fasta = config_lib.bench_path("designs", f"{cell}.fa")
    sidecar = config_lib.bench_path("results", "cells", f"{cell}.json")
    reference = Path(cluster["paths"]["pyrosetta_supercharge_script"])

    if dry_run:
        say_would(f"load mover settings from {reference} "
                  f"({'present' if reference.exists() else 'NOT PRESENT'})")
        say_would(f"read {scaffold_pdb} "
                  f"({'present' if scaffold_pdb.exists() else 'NOT PRESENT'})")
        say_would(f"cd {workdir}  (mover writes fixed-name files into the CWD)")
        say_would(f"init PyRosetta with "
                  f"'{PYROSETTA_BASE_OPTIONS} -jran {jran_for(seed)}' "
                  f"(cell seed {seed} folded into signed 32-bit range)")
        direction = "positive" if delta_q > 0 else "negative"
        say_would(f"configure Supercharge: target_net_charge={target}, "
                  f"avnapsa={avnapsa}, direction={direction} (from delta_q={delta_q:+d}), "
                  f"dont_mutate_hbonded_sidechains={hbond_protect}")
        say_would(f"apply it {nstruct} time(s), dumping each to {workdir}")
        say_would(f"write {fasta} and {sidecar}")
        return

    if not scaffold_pdb.exists():
        raise SystemExit(f"missing {scaffold_pdb}. Phase 1 must complete first.")
    workdir.mkdir(parents=True, exist_ok=True)
    for p in (fasta, sidecar):
        p.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "cell_id": cell, "scaffold_id": scaffold_id, "method": method,
        "weights": "",
        # PLAN.md Section 5 convention: hbond_filter True means h-bonded
        # sidechains were PROTECTED. On this arm that is
        # dont_mutate_hbonded_sidechains. AvNAPSA mode never consults the
        # switch, so it stays null rather than claiming a value it did not use.
        "hbond_filter": None if avnapsa else hbond_protect,
        "delta_q_density": int(task["delta_q_density"]),
        "delta_q": delta_q, "target_charge": target,
        "n_samples": nstruct, "seed": seed,
        "avnapsa": avnapsa,
        "reference_script": str(reference),
        "pyrosetta_jran": jran_for(seed),
        "pyrosetta_options": f"{PYROSETTA_BASE_OPTIONS} -jran {jran_for(seed)}",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }

    import pyrosetta                                              # noqa: PLC0415
    pyrosetta.init(options=f"{PYROSETTA_BASE_OPTIONS} -jran {jran_for(seed)}",
                   silent=True)
    from pyrosetta.rosetta.protocols.design_opt import Supercharge   # noqa: PLC0415

    original_cwd = Path.cwd()
    started = time.time()
    replicates: list[dict] = []
    try:
        wt_pose = pyrosetta.pose_from_pdb(str(scaffold_pdb))
        wt_seq = wt_pose.sequence()
        record["n_residues"] = len(wt_seq)
        record["q_wt"] = charge_lib.net_charge(wt_seq)

        os.chdir(workdir)
        for i in range(nstruct):
            work = wt_pose.clone()
            mover = Supercharge()
            applied = configure_mover(mover, target, delta_q, avnapsa,
                                      hbond_protect)
            if i == 0:
                record["mover_settings"] = applied
            mover.apply(work)
            seq = work.sequence()
            q = charge_lib.net_charge(seq)
            n_mut, muts = charge_lib.count_mutations(wt_seq, seq)
            out_pdb = workdir / f"{cell}_s{i:02d}.pdb"
            work.dump_pdb(str(out_pdb))
            replicates.append({
                "sample_index": i, "sequence": seq, "q_actual": q,
                "q_error_abs": abs(q - target), "hit_exact": q == target,
                "n_mutations": n_mut, "mutations": ";".join(muts),
                "mover_net_charge": int(mover.get_net_charge(work)),
                "pdb": str(out_pdb.relative_to(config_lib.BENCHMARK_ROOT)),
            })
        record["status"] = "ok"
        record["fail_reason"] = ""
    except Exception as exc:                                      # noqa: BLE001
        record["status"] = "failed"
        record["fail_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_seconds"] = round(time.time() - started, 2)
        os.chdir(original_cwd)

    record["replicates"] = replicates
    record["n_designs"] = len(replicates)
    if replicates:
        record["n_hit_exact"] = sum(1 for r in replicates if r["hit_exact"])
        record["n_unique_mutation_sets"] = len({r["mutations"] for r in replicates})

        # Same FASTA shape as the MPNN arms so lib/io.parse_fasta reads both.
        # No score or temperature: the mover produces neither, and inventing
        # them would put uncomputed numbers in a results file.
        with open(fasta, "w") as fh:
            fh.write(f">{scaffold_id},charge={record['q_wt']}\n{wt_seq}\n")
            for r in replicates:
                fh.write(f">{scaffold_id}_{r['sample_index']},charge={r['q_actual']}\n")
                fh.write(f"{r['sequence']}\n")
    else:
        record["n_designs"] = 0

    with open(sidecar, "w") as fh:
        json.dump(record, fh, indent=2)

    log(f"{cell}: status={record['status']} structures={record['n_designs']} "
        f"hit_exact={record.get('n_hit_exact')}/{nstruct} "
        f"unique_sets={record.get('n_unique_mutation_sets')} "
        f"{record['wall_seconds']}s -> {sidecar}")
    if record["status"] == "failed":
        log(f"  fail_reason: {record['fail_reason']}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None, method: str | None) -> None:
    import subprocess                                             # noqa: PLC0415

    task_path = config_lib.bench_path(TASK_FILE)
    tasks = read_tasks() if task_path.exists() else []
    sbatch = config_lib.bench_path("slurm", "array_baseline.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase3")
    res = cluster["slurm"]["phase3_baseline"]
    throttle = cluster["slurm"]["max_cpu_jobs"]

    if array:
        spec = array
    elif method:
        ids = sorted(int(t["task_id"]) for t in tasks if t["method"] == method)
        if not ids:
            raise SystemExit(f"no tasks for method {method!r} in {task_path}")
        # Each arm occupies a contiguous block of task ids, so collapse to a
        # range and keep the concurrency throttle. Falling back to an explicit
        # list would drop the throttle and blow past the CPU queue cap.
        if ids == list(range(ids[0], ids[-1] + 1)):
            spec = f"{ids[0]}-{ids[-1]}%{throttle}"
        else:
            spec = ",".join(str(i) for i in ids)
    elif tasks:
        spec = f"0-{len(tasks) - 1}%{throttle}"
    else:
        spec = f"0-<N-1>%{throttle}"

    cmd = ["sbatch", f"--array={spec}",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT}",
           f"--partition={cluster['slurm']['cpu_partition']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           f"--output={logdir}/%A_%a.out", f"--error={logdir}/%A_%a.out",
           str(sbatch)]

    if dry_run:
        shown = " ".join(cmd)
        if len(shown) > 400:
            shown = shown[:400] + " ...(array spec truncated)"
        say_would(f"submit: {shown}")
        if not tasks:
            say_would(f"know N once {task_path} exists (run --emit-tasks for real)")
        say_would(f"create {logdir} and log to %A_%a.out there")
        return

    if not tasks:
        raise SystemExit(f"missing {task_path}. Run --emit-tasks first.")
    logdir.mkdir(parents=True, exist_ok=True)
    log(f"submitting {len(spec.split(',')) if ',' in spec else 'range'} tasks")
    subprocess.run(cmd, check=True)

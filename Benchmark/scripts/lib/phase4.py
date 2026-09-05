"""Shared runner for Phase 4: the rejection ablation and the random control.

PLAN.md Section 7. Two experiments share one task file and one Slurm array
because they answer the same question from opposite ends:

* **Vanilla rejection sampling** draws 2,000 sequences per scaffold from
  unmodified ProteinMPNN at T = 0.3, restricted to exactly the designable
  positions the supercharging run uses, and counts how many land on each charge
  target. It measures what the base model does on its own.
* **The random-charge control** mutates randomly chosen designable positions to
  K/R or D/E until the target charge is reached. It measures what is left when
  the learned prior is removed but the charge arithmetic is kept.

Decision D, PLAN.md Section 0.2: CPU everywhere. `rejection_curve.csv` keeps the
frozen column names `gpu_seconds_per_sample`, `expected_gpu_seconds_per_hit` and
`mpnn_sc_gpu_seconds_per_hit`, and under that decision they hold CPU seconds.
Renaming them is a schema change and goes through PLAN.md first.

Restriction to the same designable set is the point of the ablation, so it is
not approximated. `lib/io.fixed_positions_for` reproduces the construction
`protein_mpnn_supercharge.py:703-706` performs, and `run_rejection` verifies
after the fact that no sample differs from wild type at a position outside that
set. A violation is a failed cell, not a warning.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from lib import charge as charge_lib
from lib import config as config_lib
from lib import io as bio

TASK_FILE = "slurm/tasks_phase4.tsv"
TASK_COLUMNS = ["task_id", "kind", "scaffold_id", "chain", "method",
                "n_residues", "q_wt", "n_samples", "seed"]

KINDS = ("rejection", "random_control")

# PLAN.md Section 7 freezes these column names. See the module docstring.
REJECTION_CURVE_COLUMNS = [
    "scaffold_id", "delta_q_density", "target_charge", "n_samples_drawn",
    "n_on_target", "hit_rate", "expected_samples_per_hit",
    "gpu_seconds_per_sample", "expected_gpu_seconds_per_hit",
    "mpnn_sc_gpu_seconds_per_hit",
]

# Not in PLAN.md's frozen list. Section 7 also asks for the empirical charge
# distribution per scaffold, which is per-scaffold rather than per-target and so
# has nowhere to live in the curve. Additive, not a schema change.
REJECTION_DISTRIBUTION_COLUMNS = [
    "scaffold_id", "n_residues", "n_designable", "q_wt", "n_samples_drawn",
    "q_mean", "q_sd", "q_min", "q_max", "seconds_per_sample", "status",
]


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


def primary_cache(scaffold_id: str) -> Path:
    """The LayerSelector cache both Phase 4 experiments read.

    The primary arm's working directory, `data/`, not the control arm's. Its
    cache was built with -mhbond, which is the configuration every headline
    number in this benchmark comes from, so it is the designable set the
    ablation has to match.
    """
    return config_lib.bench_path("data", "parsed", f"{scaffold_id}_seq_indices.pkl")


# ---------------------------------------------------------------------------
# task file
# ---------------------------------------------------------------------------

def emit_tasks(cfg: dict, dry_run: bool) -> None:
    """One task per scaffold per experiment, both experiments in one file.

    Granularity is the scaffold, not the cell. A rejection task draws one pool
    of 2,000 samples that every target for that scaffold is then counted
    against, and a random-control task walks that scaffold's 8 targets in a
    fraction of a second, so a per-cell array would be 200 jobs of scheduling
    overhead for no rerun benefit.
    """
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    out = config_lib.bench_path(TASK_FILE)
    resolved = cfg.get("resolved_targets")
    phase4 = cfg["phase4"]
    rej, rnd = phase4["rejection"], phase4["random_control"]

    if dry_run:
        say_would(f"read {manifest_path} "
                  f"({'present' if manifest_path.exists() else 'NOT PRESENT'})")
        say_would("read resolved_targets from config/benchmark.yaml "
                  f"({'present' if resolved else 'NOT PRESENT'})")
        say_would(f"  kind rejection: all scaffolds x {rej['n_samples']} samples at "
                  f"T={rej['temperature']}, {rej['weights']} weights, {rej['model']}")
        say_would(f"  kind random_control: all scaffolds x "
                  f"{len(cfg['charge_ladder']['delta_q_density'])} targets x "
                  f"{rnd['n_replicates']} replicates")
        say_would(f"write {out} with columns {TASK_COLUMNS}")
        return

    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}. Phase 1 must complete first.")
    if not resolved:
        raise SystemExit("config/benchmark.yaml has no resolved_targets block.")
    if rej["n_samples"] % rej["batch_size"]:
        raise SystemExit(
            f"phase4.rejection: n_samples={rej['n_samples']} is not divisible by "
            f"batch_size={rej['batch_size']}. protein_mpnn_run.py computes "
            "NUM_BATCHES with integer division, so the remainder would be dropped "
            "and the pool would be smaller than the recorded n_samples_drawn."
        )

    manifest = read_csv(manifest_path)
    by_id = {r["scaffold_id"]: r for r in manifest}
    base_seed = cfg["seeds"]["base_seed"]
    rows, task_id = [], 0

    for kind, arm in (("rejection", rej), ("random_control", rnd)):
        n_samples = rej["n_samples"] if kind == "rejection" else rnd["n_replicates"]
        for scaffold_id in sorted(by_id):
            if scaffold_id not in resolved:
                raise SystemExit(f"{scaffold_id} has no resolved targets.")
            rows.append({
                "task_id": task_id,
                "kind": kind,
                "scaffold_id": scaffold_id,
                "chain": by_id[scaffold_id]["chain"],
                "method": arm["name"],
                "n_residues": by_id[scaffold_id]["n_residues"],
                "q_wt": by_id[scaffold_id]["q_wt"],
                "n_samples": n_samples,
                # Target charge is not part of the key: a rejection task covers
                # every target at once, and a random-control task derives a
                # separate per-cell seed for each target below.
                "seed": bio.cell_seed(base_seed, scaffold_id, arm["name"], 0),
            })
            task_id += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TASK_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    log(f"wrote {out} with {len(rows)} tasks (array indices 0 to {len(rows) - 1})")
    for kind in KINDS:
        ids = [r["task_id"] for r in rows if r["kind"] == kind]
        log(f"  {kind}: {len(ids)} tasks, indices {ids[0]} to {ids[-1]}")


def _task_for(task_index: int, kind: str) -> dict:
    tasks = read_tasks()
    task = next((t for t in tasks
                 if int(t["task_id"]) == task_index and t["kind"] == kind), None)
    if task is None:
        raise SystemExit(
            f"no task with task_id={task_index} and kind={kind!r} in {TASK_FILE}")
    return task


def _slurm_fields() -> dict:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "gpus": os.environ.get("SLURM_GPUS_ON_NODE", ""),
    }


# ---------------------------------------------------------------------------
# Vanilla rejection sampling
# ---------------------------------------------------------------------------

def parse_vanilla_fasta(path: Path, wt_seq: str) -> list[str]:
    """Read protein_mpnn_run.py's output FASTA into a list of design sequences.

    Not `lib/io.parse_fasta`: that parses the supercharge script's
    `key=value` headers, which carry `charge=`. The stock runner writes a
    different header (`>T=0.3, sample=1, score=..., seq_recovery=...`) and no
    charge at all, so charges are computed here through `lib/charge.py` rather
    than read from a header that does not exist.

    The first record is the native sequence (protein_mpnn_run.py:400). That is
    checked against the cached wild type instead of assumed, because if the two
    disagree the run was restricted against the wrong structure and every
    downstream count would be wrong.
    """
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))

    if not records:
        raise ValueError(f"{path}: no records parsed")
    native_seq = records[0][1]
    if native_seq != wt_seq:
        raise ValueError(
            f"{path}: the native record does not match the cached wild type "
            f"(runner {len(native_seq)} aa, cache {len(wt_seq)} aa). The sampling "
            "ran against a different structure than the designable set came from."
        )
    designs = [seq for _, seq in records[1:]]
    for seq in designs:
        if "/" in seq:
            raise ValueError(
                f"{path}: a sampled sequence contains a chain break '/'. The "
                "scaffold was expected to hold exactly one chain.")
        if len(seq) != len(wt_seq):
            raise ValueError(
                f"{path}: sampled sequence of length {len(seq)}, expected {len(wt_seq)}")
    return designs


def _off_target_positions(wt_seq: str, seq: str, designable: set[int]) -> list[int]:
    """1-based positions where a design differs from wild type outside the
    designable set. Must be empty; see the module docstring."""
    return [i for i, (a, b) in enumerate(zip(wt_seq, seq), start=1)
            if a != b and i not in designable]


def build_mpnn_argv(cfg: dict, cluster: dict, task: dict, pdb: Path,
                    fixed_jsonl: Path, out_folder: Path) -> list[str]:
    """The command line for one scaffold's vanilla sampling pool.

    `--use_soluble_model` rather than `--path_to_model_weights`: passing an
    explicit weights folder overrides the soluble/vanilla choice inside
    protein_mpnn_run.py, and the primary arm's checkpoint is
    `soluble_model_weights/v_48_020.pt`, which is exactly what the flag selects
    from the clone. Same file, chosen the way the runner expects.
    """
    rej = cfg["phase4"]["rejection"]
    runner = Path(cluster["paths"]["proteinmpnn"]) / "protein_mpnn_run.py"
    argv = [
        sys.executable, str(runner),
        "--pdb_path", str(pdb),
        "--pdb_path_chains", str(task["chain"]),
        "--fixed_positions_jsonl", str(fixed_jsonl),
        "--out_folder", str(out_folder),
        "--num_seq_per_target", str(rej["n_samples"]),
        "--batch_size", str(rej["batch_size"]),
        "--sampling_temp", str(rej["temperature"]),
        "--model_name", str(rej["model"]),
        "--seed", str(int(task["seed"])),
    ]
    if rej["weights"] == "soluble":
        argv.append("--use_soluble_model")
    elif rej["weights"] != "original":
        raise SystemExit(f"phase4.rejection.weights must be 'soluble' or 'original', "
                         f"got {rej['weights']!r}")
    return argv


def run_rejection(cfg: dict, cluster: dict, task_index: int, dry_run: bool) -> None:
    task = _task_for(task_index, "rejection")
    scaffold_id = task["scaffold_id"]
    rej = cfg["phase4"]["rejection"]
    seed = int(task["seed"])

    pdb = config_lib.bench_path("data", "scaffolds", scaffold_id, f"{scaffold_id}.pdb")
    cache = primary_cache(scaffold_id)
    fixed_jsonl = config_lib.bench_path("data", "parsed",
                                        f"{scaffold_id}_fixed_positions.jsonl")
    out_folder = config_lib.bench_path(rej["out_dir"], scaffold_id)
    fasta = out_folder / "seqs" / f"{scaffold_id}.fa"
    result = config_lib.bench_path("results", "rejection", f"{scaffold_id}.json")

    if dry_run:
        say_would(f"read {cache} "
                  f"({'present' if cache.exists() else 'NOT PRESENT, Phase 1 first'})")
        say_would(f"write {fixed_jsonl}, the complement of that cache's designable "
                  "set, in protein_mpnn_run.py's fixed-positions format")
        say_would(f"read {pdb} ({'present' if pdb.exists() else 'NOT PRESENT'})")
        argv = build_mpnn_argv(cfg, cluster, task, pdb, fixed_jsonl, out_folder)
        say_would(f"run, with {rej['n_samples'] // rej['batch_size']} batches of "
                  f"{rej['batch_size']}:")
        print("      " + " \\\n        ".join(
            " ".join(argv[i:i + 2]) for i in range(0, len(argv), 2)), flush=True)
        say_would(f"parse {fasta}, compute every sample's charge through lib/charge.py")
        say_would("verify no sample differs from wild type outside the designable set")
        say_would(f"write {result} with per-sample charges, mutation counts and timing")
        return

    if not cache.exists():
        raise SystemExit(f"missing {cache}. Phase 1 must complete for the primary arm.")
    if not pdb.exists():
        raise SystemExit(f"missing {pdb}. Phase 1 must complete first.")

    wt_seq, designable, q_wt_cache = bio.load_designable_cache(cache)
    bio.write_fixed_positions_jsonl(
        fixed_jsonl, bio.fixed_positions_for(scaffold_id, str(task["chain"]),
                                             wt_seq, designable))

    record = {
        "scaffold_id": scaffold_id,
        "kind": "rejection",
        "method": task["method"],
        "weights": rej["weights"],
        "model": rej["model"],
        "temperature": rej["temperature"],
        "n_samples_requested": int(rej["n_samples"]),
        "batch_size": int(rej["batch_size"]),
        "seed": seed,
        "n_residues": len(wt_seq),
        "n_designable": len(designable),
        "q_wt": q_wt_cache,
        "fixed_positions_jsonl": str(fixed_jsonl.relative_to(config_lib.BENCHMARK_ROOT)),
        "fasta": str(fasta.relative_to(config_lib.BENCHMARK_ROOT)),
        **_slurm_fields(),
    }
    # The cache's own q_wt is recomputed rather than trusted, so the charge in
    # this record and the charges of the samples come from the same function.
    record["q_wt_recomputed"] = charge_lib.net_charge(wt_seq)

    out_folder.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    argv = build_mpnn_argv(cfg, cluster, task, pdb, fixed_jsonl, out_folder)
    record["argv"] = argv

    started = time.time()
    try:
        if record["q_wt_recomputed"] != q_wt_cache:
            raise ValueError(
                f"cache q_wt {q_wt_cache} disagrees with lib/charge.py "
                f"{record['q_wt_recomputed']} for {scaffold_id}")
        proc = subprocess.run(argv, capture_output=True, text=True)
        record["returncode"] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError(
                f"protein_mpnn_run.py exited {proc.returncode}: "
                f"{proc.stderr.strip()[-500:]}")
        designs = parse_vanilla_fasta(fasta, wt_seq)
        if len(designs) != rej["n_samples"]:
            raise ValueError(
                f"{fasta}: {len(designs)} sampled sequences, expected "
                f"{rej['n_samples']}")

        designable_set = set(designable)
        charges, mutation_counts = [], []
        for seq in designs:
            off = _off_target_positions(wt_seq, seq, designable_set)
            if off:
                raise ValueError(
                    f"a sample differs from wild type at non-designable positions "
                    f"{off[:10]}. The fixed-positions restriction did not take, so "
                    "this pool is not comparable to the supercharging runs.")
            charges.append(charge_lib.net_charge(seq))
            mutation_counts.append(charge_lib.count_mutations(wt_seq, seq)[0])

        record["charges"] = charges
        record["n_mutations"] = mutation_counts
        record["n_samples_drawn"] = len(charges)
        record["q_mean"] = round(statistics.fmean(charges), 4)
        record["q_sd"] = round(statistics.stdev(charges), 4) if len(charges) > 1 else 0.0
        record["q_min"] = min(charges)
        record["q_max"] = max(charges)
        record["status"] = "ok"
        record["fail_reason"] = ""
    except Exception as exc:                                      # noqa: BLE001
        record["status"] = "failed"
        record["fail_reason"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("n_samples_drawn", 0)
    finally:
        record["wall_seconds"] = round(time.time() - started, 2)

    if record["status"] == "ok":
        record["seconds_per_sample"] = round(
            record["wall_seconds"] / record["n_samples_drawn"], 5)
    else:
        record["seconds_per_sample"] = None

    with open(result, "w") as fh:
        json.dump(record, fh, indent=2)

    log(f"{scaffold_id} rejection: status={record['status']} "
        f"samples={record['n_samples_drawn']} "
        f"q_mean={record.get('q_mean')} q_sd={record.get('q_sd')} "
        f"{record['wall_seconds']}s -> {result}")
    if record["status"] == "failed":
        log(f"  fail_reason: {record['fail_reason']}")


# ---------------------------------------------------------------------------
# Random-charge negative control
# ---------------------------------------------------------------------------

POSITIVE_CHOICES = ("K", "R")
NEGATIVE_CHOICES = ("D", "E")


def random_charge_design(wt_seq: str, designable: list[int], delta_q: int,
                         rng: random.Random) -> tuple[str, dict]:
    """Mutate randomly chosen designable positions until the target is reached.

    PLAN.md Section 7: "pick designable positions uniformly at random and mutate
    to K or R (positive) or D or E (negative) until the target is reached."

    Three details that the sentence leaves open, all resolved so the control
    measures randomness rather than an artefact of the sampler:

    1. **Positions already carrying the target polarity are skipped.** Turning a
       K into an R moves no charge, so accepting such a pick would make the
       number of mutations depend on the wild-type composition rather than on
       the charge that had to be moved.
    2. **A pick that would overshoot is skipped, not accepted.** Reversing an
       Asp costs 2 units of charge while converting a neutral residue costs 1,
       so when one unit remains, only a neutral position can land the target
       exactly. Accepting the overshoot instead would miss by one in a large
       fraction of cells for reasons that have nothing to do with the random
       prior, and would flatter the learned model in the comparison.
    3. **Gly, Pro and Cys need no special handling**: the cached designable set
       was built with mutate_glyprocys False and already excludes them.

    The walk is over a random permutation of the designable positions, so a
    position is considered at most once and the pass terminates. If the pass
    ends with charge left to move, the target is unreachable under this
    scaffold's designable set and the design is returned short. That is a
    result, not an error.

    Returns (sequence, diagnostics).
    """
    if delta_q == 0:
        raise ValueError("delta_q == 0 leaves nothing for the control to do")

    raise_charge = delta_q > 0
    choices = POSITIVE_CHOICES if raise_charge else NEGATIVE_CHOICES
    reversible = charge_lib.NEGATIVE_RESIDUES if raise_charge else charge_lib.POSITIVE_RESIDUES

    seq = list(wt_seq)
    remaining = abs(delta_q)
    order = list(designable)
    rng.shuffle(order)

    n_considered = n_skipped_same_polarity = n_skipped_overshoot = 0
    for pos in order:
        if remaining == 0:
            break
        n_considered += 1
        wt_aa = seq[pos - 1]
        if wt_aa in choices:
            n_skipped_same_polarity += 1
            continue
        step = 2 if wt_aa in reversible else 1
        if step > remaining:
            n_skipped_overshoot += 1
            continue
        seq[pos - 1] = rng.choice(choices)
        remaining -= step

    return "".join(seq), {
        "positions_considered": n_considered,
        "skipped_same_polarity": n_skipped_same_polarity,
        "skipped_would_overshoot": n_skipped_overshoot,
        "charge_short_by": remaining,
        "exhausted_designable": remaining > 0,
    }


def run_random_control(cfg: dict, cluster: dict, task_index: int,
                       dry_run: bool) -> None:
    """One task is one scaffold: all of its targets, all replicates."""
    task = _task_for(task_index, "random_control")
    scaffold_id = task["scaffold_id"]
    method = task["method"]
    n_replicates = int(task["n_samples"])
    base_seed = cfg["seeds"]["base_seed"]
    targets = cfg["resolved_targets"][scaffold_id]
    cache = primary_cache(scaffold_id)

    if dry_run:
        say_would(f"read {cache} "
                  f"({'present' if cache.exists() else 'NOT PRESENT, Phase 1 first'})")
        say_would(f"for each of {len(targets)} targets, build {n_replicates} designs by "
                  "mutating randomly chosen designable positions to K/R or D/E")
        for cell in targets:
            name = bio.cell_id(scaffold_id, method, cell["target_charge"])
            say_would(f"  {name}: delta_q={cell['delta_q']:+d}, seed="
                      f"{bio.cell_seed(base_seed, scaffold_id, method, cell['target_charge'])}"
                      f" -> designs/{name}.fa and results/cells/{name}.json")
        return

    if not cache.exists():
        raise SystemExit(f"missing {cache}. Phase 1 must complete for the primary arm.")

    wt_seq, designable, q_wt_cache = bio.load_designable_cache(cache)
    q_wt = charge_lib.net_charge(wt_seq)
    if q_wt != q_wt_cache:
        raise SystemExit(
            f"{scaffold_id}: cache q_wt {q_wt_cache} disagrees with lib/charge.py {q_wt}")

    for cell in targets:
        target = int(cell["target_charge"])
        delta_q = int(cell["delta_q"])
        cell_name = bio.cell_id(scaffold_id, method, target)
        seed = bio.cell_seed(base_seed, scaffold_id, method, target)
        fasta = config_lib.bench_path("designs", f"{cell_name}.fa")
        sidecar = config_lib.bench_path("results", "cells", f"{cell_name}.json")
        for p in (fasta, sidecar):
            p.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "cell_id": cell_name, "scaffold_id": scaffold_id, "method": method,
            "weights": "", "hbond_filter": None,
            "delta_q_density": int(cell["delta_q_density"]),
            "delta_q": delta_q, "target_charge": target,
            "n_samples": n_replicates, "seed": seed,
            "n_residues": len(wt_seq), "n_designable": len(designable),
            "q_wt": q_wt,
            **_slurm_fields(),
        }

        started = time.time()
        replicates: list[dict] = []
        try:
            for i in range(n_replicates):
                # A per-replicate stream derived from the cell seed, so rerunning
                # one cell reproduces all ten and the replicates stay independent.
                rng = random.Random(f"{seed}|{i}")
                seq, diag = random_charge_design(wt_seq, designable, delta_q, rng)
                q = charge_lib.net_charge(seq)
                expected = q_wt + (delta_q - (diag["charge_short_by"]
                                              * (1 if delta_q > 0 else -1)))
                if q != expected:
                    raise ValueError(
                        f"{cell_name} replicate {i}: charge bookkeeping disagrees "
                        f"with lib/charge.py ({q} vs {expected})")
                n_mut, muts = charge_lib.count_mutations(wt_seq, seq)
                replicates.append({
                    "sample_index": i, "sequence": seq, "q_actual": q,
                    "q_error_abs": abs(q - target), "hit_exact": q == target,
                    "n_mutations": n_mut, "mutations": ";".join(muts),
                    **diag,
                })
            record["status"] = "ok"
            record["fail_reason"] = ""
        except Exception as exc:                                  # noqa: BLE001
            record["status"] = "failed"
            record["fail_reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["wall_seconds"] = round(time.time() - started, 4)

        record["replicates"] = replicates
        record["n_designs"] = len(replicates)
        if replicates:
            record["n_hit_exact"] = sum(1 for r in replicates if r["hit_exact"])
            record["n_unique_mutation_sets"] = len({r["mutations"] for r in replicates})
            record["n_exhausted_designable"] = sum(
                1 for r in replicates if r["exhausted_designable"])
            # Same FASTA shape as every other arm so lib/io.parse_fasta reads it.
            # No score and no temperature: this arm computes neither, and filling
            # them would put uncomputed numbers in a results file.
            with open(fasta, "w") as fh:
                fh.write(f">{scaffold_id},charge={q_wt}\n{wt_seq}\n")
                for r in replicates:
                    fh.write(f">{scaffold_id}_{r['sample_index']},"
                             f"charge={r['q_actual']}\n{r['sequence']}\n")

        with open(sidecar, "w") as fh:
            json.dump(record, fh, indent=2)

        log(f"{cell_name}: status={record['status']} designs={record['n_designs']} "
            f"hit_exact={record.get('n_hit_exact')}/{n_replicates} "
            f"unique_sets={record.get('n_unique_mutation_sets')} "
            f"short={record.get('n_exhausted_designable')} -> {sidecar}")
        if record["status"] == "failed":
            log(f"  fail_reason: {record['fail_reason']}")


# ---------------------------------------------------------------------------
# results/rejection_curve.csv
# ---------------------------------------------------------------------------

def _mpnn_seconds_per_hit(scaffold_id: str, target: int) -> str:
    """Cost of one on-target design from the primary supercharging arm.

    Read from the Phase 2 cell: its wall time divided by the number of its
    designs that landed on target. The hit count is recounted from the FASTA
    through lib/charge.py rather than taken from a header, and the cell's own
    sidecar supplies the wall time. Returns "" when the cell is missing, and a
    censored ">x" when the cell produced no on-target design, because dividing
    by zero there would manufacture an infinity that reads as a measurement.
    """
    cell = bio.cell_id(scaffold_id, "mpnn_soluble", target)
    sidecar = config_lib.bench_path("results", "cells", f"{cell}.json")
    fasta = config_lib.bench_path("designs", f"{cell}.fa")
    if not sidecar.exists() or not fasta.exists():
        return ""
    with open(sidecar) as fh:
        rec = json.load(fh)
    if rec.get("status") != "ok":
        return ""
    _, designs = bio.split_native(bio.parse_fasta(fasta))
    n_hit = sum(1 for d in designs if charge_lib.net_charge(d.sequence) == target)
    wall = float(rec["wall_seconds"])
    if n_hit == 0:
        return f">{wall:.4f}"
    return f"{wall / n_hit:.4f}"


def build_curve(cfg: dict, dry_run: bool) -> None:
    """Write results/rejection_curve.csv and the per-scaffold distribution.

    Reads only files on disk: the per-scaffold sample pools written by
    run_rejection, and the Phase 2 primary-arm cells for the comparison column.
    A scaffold whose pool is missing or failed is skipped and reported, never
    filled in.
    """
    resolved = cfg["resolved_targets"]
    curve_path = config_lib.bench_path("results", "rejection_curve.csv")
    dist_path = config_lib.bench_path("results", "rejection_distribution.csv")
    pool_dir = config_lib.bench_path("results", "rejection")

    if dry_run:
        say_would(f"read every {pool_dir}/<scaffold>.json")
        say_would(f"count, per target, how many of the pool's samples equal the target "
                  f"charge, and write {curve_path} with columns "
                  f"{REJECTION_CURVE_COLUMNS}")
        say_would("report expected_samples_per_hit as censored '>N' where no sample "
                  "landed on target, rather than extrapolating a rate")
        say_would(f"write {dist_path} with the per-scaffold empirical charge "
                  "distribution (PLAN.md Section 7)")
        return

    curve_rows, dist_rows, missing = [], [], []
    for scaffold_id in sorted(resolved):
        pool_file = pool_dir / f"{scaffold_id}.json"
        if not pool_file.exists():
            missing.append((scaffold_id, "no pool file"))
            continue
        with open(pool_file) as fh:
            pool = json.load(fh)
        if pool.get("status") != "ok":
            missing.append((scaffold_id, f"pool status={pool.get('status')}"))
            dist_rows.append({"scaffold_id": scaffold_id,
                              "n_residues": pool.get("n_residues", ""),
                              "n_designable": pool.get("n_designable", ""),
                              "q_wt": pool.get("q_wt", ""),
                              "n_samples_drawn": pool.get("n_samples_drawn", 0),
                              "status": pool.get("status", "failed")})
            continue

        charges = pool["charges"]
        n_drawn = len(charges)
        sec_per_sample = pool["seconds_per_sample"]
        dist_rows.append({
            "scaffold_id": scaffold_id, "n_residues": pool["n_residues"],
            "n_designable": pool["n_designable"], "q_wt": pool["q_wt"],
            "n_samples_drawn": n_drawn, "q_mean": pool["q_mean"],
            "q_sd": pool["q_sd"], "q_min": pool["q_min"], "q_max": pool["q_max"],
            "seconds_per_sample": sec_per_sample, "status": "ok",
        })

        for cell in resolved[scaffold_id]:
            target = int(cell["target_charge"])
            n_on_target = sum(1 for q in charges if q == target)
            if n_on_target:
                hit_rate = n_on_target / n_drawn
                per_hit = f"{1.0 / hit_rate:.4f}"
                sec_per_hit = f"{sec_per_sample / hit_rate:.4f}"
            else:
                # PLAN.md Section 7: censored, not extrapolated.
                hit_rate = 0.0
                per_hit = f">{n_drawn}"
                sec_per_hit = f">{sec_per_sample * n_drawn:.4f}"
            curve_rows.append({
                "scaffold_id": scaffold_id,
                "delta_q_density": cell["delta_q_density"],
                "target_charge": target,
                "n_samples_drawn": n_drawn,
                "n_on_target": n_on_target,
                "hit_rate": f"{hit_rate:.6f}",
                "expected_samples_per_hit": per_hit,
                "gpu_seconds_per_sample": sec_per_sample,
                "expected_gpu_seconds_per_hit": sec_per_hit,
                "mpnn_sc_gpu_seconds_per_hit": _mpnn_seconds_per_hit(scaffold_id, target),
            })

    bio.write_csv(curve_path, curve_rows, REJECTION_CURVE_COLUMNS)
    bio.write_csv(dist_path, dist_rows, REJECTION_DISTRIBUTION_COLUMNS)
    log(f"wrote {curve_path} with {len(curve_rows)} rows "
        f"({len({r['scaffold_id'] for r in curve_rows})} scaffolds)")
    log(f"wrote {dist_path} with {len(dist_rows)} rows")
    if missing:
        log(f"  {len(missing)} scaffolds contributed no curve rows:")
        for scaffold_id, why in missing:
            log(f"    {scaffold_id}: {why}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit(cluster: dict, dry_run: bool, array: str | None, kind: str,
           resource_key: str) -> None:
    task_path = config_lib.bench_path(TASK_FILE)
    tasks = read_tasks() if task_path.exists() else []
    sbatch = config_lib.bench_path("slurm", "array_phase4.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase4")
    res = cluster["slurm"][resource_key]
    throttle = cluster["slurm"]["max_cpu_jobs"]

    if array:
        spec = array
    elif tasks:
        ids = sorted(int(t["task_id"]) for t in tasks if t["kind"] == kind)
        if not ids:
            raise SystemExit(f"no tasks of kind {kind!r} in {task_path}")
        # Each kind occupies a contiguous block, so this stays a range and keeps
        # the concurrency throttle rather than degrading to an explicit list.
        if ids == list(range(ids[0], ids[-1] + 1)):
            spec = f"{ids[0]}-{ids[-1]}%{throttle}"
        else:
            spec = ",".join(str(i) for i in ids)
    else:
        spec = f"<{kind} block>%{throttle}"

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
        if not tasks:
            say_would(f"know the index block once {task_path} exists "
                      "(run --emit-tasks for real)")
        say_would(f"create {logdir} and log to %A_%a.out there")
        if kind == "rejection":
            say_would("NOTE: PLAN.md Section 2.1 requires timing one scaffold on CPU "
                      "before launching this array. Run --time-one first and set "
                      f"slurm.{resource_key}.time from it.")
        return

    if not tasks:
        raise SystemExit(f"missing {task_path}. Run --emit-tasks first.")
    logdir.mkdir(parents=True, exist_ok=True)
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def time_one(cluster: dict, dry_run: bool, task_index: int,
             resource_key: str) -> None:
    """Submit a single task so its walltime can size the array."""
    submit(cluster, dry_run, str(task_index), "rejection", resource_key)

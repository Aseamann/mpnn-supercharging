#!/usr/bin/env python3
"""Phase 4c: mutational efficiency of UNGUIDED ProteinMPNN, from the Phase 4a pools.

Added 2026-09-05. F4 compares mutations per unit charge across the supercharging
methods. The question it could not answer was what the base model costs when it
is not steered at all: `vanilla_rejection` is a pool of 2,000 unguided samples
per scaffold, and the samples in that pool that happen to land on a ladder rung
are exactly "what unguided ProteinMPNN spends to reach this charge".

Those samples exist and were never scored. `results/rejection_curve.csv` counts
them (6,428 on-target samples across 81 of 200 cells) but records nothing about
the sequences. This script reads the pools and writes one row per on-target
sample so F4 can carry an unguided series next to the guided ones.

Two properties of that series matter and are recorded rather than described:

  * It is missing at the two most positive rungs. Not thinned, missing. The
    pools contain zero on-target samples at +16 and +24 across all 25 scaffolds,
    so those bins have no unguided box at all. The negative rungs all have
    samples, 236 at -24 and 314 at -16, so the gap is one-sided: the base model
    leans negative and cannot be pushed positive by luck. That is the same fact
    F2 and F3 report from the other direction.
  * It is survivorship-filtered. A cell contributes only where the base model
    reached the target at least once, so the unguided cost is measured on the
    easiest scaffold-target combinations only and is a lower bound on what the
    unguided model would need across the full ladder.

Nothing here recomputes charge: `lib/charge.py` is the single source of truth,
per CLAUDE.md, and the target for each rung comes from the same
`resolved_targets` block in config/benchmark.yaml that the design arms used.

No cluster work. Reading 25 pools of 2,000 short sequences is seconds of pure
Python, but it is still submitted through sbatch rather than run in an
orchestration shell, for the same reason every other phase is.

Usage:
  12_rejection_hit_metrics.py --run [--dry-run]
  12_rejection_hit_metrics.py --submit [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import charge as charge_lib   # noqa: E402
from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402
from lib import phase4                 # noqa: E402

OUT_CSV = "results/rejection_hits.csv"
COLUMNS = ["scaffold_id", "n_residues", "n_designable", "q_wt",
           "delta_q_density", "target_charge", "sample_index", "q_actual",
           "n_mutations", "mutations", "mut_per_charge",
           "n_samples_drawn", "n_on_target_in_cell"]


def log(msg: str) -> None:
    print(msg, flush=True)


def say_would(msg: str) -> None:
    print(f"[dry-run] would {msg}", flush=True)


def pool_path(cfg: dict, scaffold_id: str) -> Path:
    out_dir = cfg["phase4"]["rejection"]["out_dir"]
    return config_lib.bench_path(out_dir, scaffold_id, "seqs", f"{scaffold_id}.fa")


def run(cfg: dict, dry_run: bool) -> None:
    resolved = cfg.get("resolved_targets")
    if not resolved:
        raise SystemExit("config/benchmark.yaml has no resolved_targets block.")
    add_his = bool(cfg["charge_definition"]["add_histidine"])
    out = config_lib.bench_path(OUT_CSV)
    scaffolds = sorted(resolved)

    if dry_run:
        present = sum(1 for s in scaffolds if pool_path(cfg, s).exists())
        say_would(f"read {len(scaffolds)} Phase 4a pools from "
                  f"{cfg['phase4']['rejection']['out_dir']}/ ({present} present)")
        say_would("read the matching data/parsed/<scaffold>_seq_indices.pkl for "
                  "the wild-type sequence and designable set")
        say_would("keep only samples whose net charge equals a resolved ladder "
                  "target exactly, the same hit_exact test the design arms use")
        say_would(f"write {out} with columns {COLUMNS}")
        return

    rows: list[dict] = []
    missing: list[str] = []
    for scaffold_id in scaffolds:
        fa = pool_path(cfg, scaffold_id)
        pkl = config_lib.bench_path("data", "parsed",
                                    f"{scaffold_id}_seq_indices.pkl")
        if not fa.exists() or not pkl.exists():
            missing.append(scaffold_id)
            continue

        wt_seq, designable, q_wt_cached = bio.load_designable_cache(pkl)
        # The stock protein_mpnn_run.py header carries no charge field, so
        # lib/io.parse_fasta rejects these pools. lib/phase4 already has the
        # parser Phase 4a used, including its check that the pool's native
        # record matches the cached wild type, so the hit counts here are read
        # off exactly the sequences rejection_curve.csv counted.
        samples = phase4.parse_vanilla_fasta(fa, wt_seq)
        q_wt = charge_lib.net_charge(wt_seq, add_his)
        if q_wt != q_wt_cached:
            raise SystemExit(
                f"{scaffold_id}: net charge {q_wt} disagrees with the cached "
                f"q_wt {q_wt_cached}.")

        # Charge every sample once, then bucket by target. A sample that lands
        # on two rungs is impossible: the rungs resolve to distinct charges.
        charged = [(i, seq, charge_lib.net_charge(seq, add_his))
                   for i, seq in enumerate(samples, start=1)]

        for cell in resolved[scaffold_id]:
            target = int(cell["target_charge"])
            hits = [(i, seq, q) for i, seq, q in charged if q == target]
            for i, seq, q in hits:
                n_mut, muts = charge_lib.count_mutations(wt_seq, seq)
                rows.append({
                    "scaffold_id": scaffold_id,
                    "n_residues": len(wt_seq),
                    "n_designable": len(designable),
                    "q_wt": q_wt,
                    "delta_q_density": cell["delta_q_density"],
                    "target_charge": target,
                    "sample_index": i,
                    "q_actual": q,
                    "n_mutations": n_mut,
                    "mutations": ";".join(muts),
                    "mut_per_charge": charge_lib.mut_per_charge(n_mut, q, q_wt),
                    "n_samples_drawn": len(samples),
                    "n_on_target_in_cell": len(hits),
                })

    if missing:
        raise SystemExit(
            f"missing pool or designable cache for {len(missing)} scaffolds: "
            f"{missing[:5]}. Phase 4a must complete first; this script does not "
            f"substitute anything for an absent pool.")

    bio.write_csv(out, rows, COLUMNS)
    log(f"wrote {out} with {len(rows)} on-target unguided samples")
    by_rung: dict[int, int] = {}
    for r in rows:
        by_rung[int(r["delta_q_density"])] = by_rung.get(int(r["delta_q_density"]), 0) + 1
    for rung in sorted(by_rung):
        log(f"  delta_q_density {rung:+d}: {by_rung[rung]} samples")
    for rung in sorted(int(c["delta_q_density"]) for s in scaffolds
                       for c in resolved[s]):
        if rung not in by_rung:
            log(f"  delta_q_density {rung:+d}: 0 samples, no unguided series")
            by_rung[rung] = 0


def submit(cluster: dict, dry_run: bool) -> None:
    sbatch = config_lib.bench_path("slurm", "phase4c_rejection_hits.sbatch")
    logdir = config_lib.bench_path("logs", "slurm", "phase4c")
    res = cluster["slurm"]["phase4_random"]
    cmd = ["sbatch",
           f"--export=BM_ROOT={config_lib.BENCHMARK_ROOT}",
           f"--partition={cluster['slurm']['cpu_partition']}",
           f"--cpus-per-task={res['cpus_per_task']}", f"--mem={res['mem']}",
           f"--time={res['time']}",
           f"--output={logdir}/%j.out", f"--error={logdir}/%j.out",
           str(sbatch)]
    if dry_run:
        say_would(f"submit: {' '.join(cmd)}")
        return
    logdir.mkdir(parents=True, exist_ok=True)
    log(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Do the work here")
    group.add_argument("--submit", action="store_true",
                       help="Submit it as a single Slurm job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be read, written or submitted")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    if args.run:
        run(cfg, args.dry_run)
    else:
        submit(config_lib.load_cluster(), args.dry_run)


if __name__ == "__main__":
    main()

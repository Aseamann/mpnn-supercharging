#!/usr/bin/env python3
"""Phase 4a: vanilla ProteinMPNN rejection sampling.

Draws 2,000 sequences per scaffold from the unmodified `protein_mpnn_run.py` at
T = 0.3, restricted to exactly the designable positions the supercharging run
uses, and counts how many land on each charge target. PLAN.md Section 7. All the
work lives in `lib/phase4.py`; this is the CLI for the rejection arm.

The restriction comes from the Phase 1 LayerSelector cache for the primary arm,
converted to `protein_mpnn_run.py`'s fixed-positions JSONL by
`lib/io.fixed_positions_for`. Without it the vanilla sampler would be solving a
different design problem and the ablation would prove nothing.

Decision D (PLAN.md Section 0.2): this runs on CPU, and the `gpu_seconds_*`
columns of `results/rejection_curve.csv` hold CPU seconds under their frozen
names.

--emit-tasks writes the task file for BOTH Phase 4 experiments, so it only needs
running once whichever script invokes it.

The temperature sweep (--sweep, PLAN.md Section 0.1 item 23) redraws the same
pools at T = 0.6 and T = 0.9 so F1 can show which ladder rungs unguided sampling
reaches when the temperature is raised. It uses a second task file, a second
sbatch script and suffixed output paths, so nothing written at T = 0.3 is
touched. Seeds are copied from the T = 0.3 rows, making temperature the only
difference between the three pools.

Usage:
  05_run_vanilla_rejection.py --emit-tasks [--dry-run]
  05_run_vanilla_rejection.py --time-one [TASK_ID] [--dry-run]
  05_run_vanilla_rejection.py --task 0 [--dry-run]
  05_run_vanilla_rejection.py --submit [--array=3,7] [--dry-run]
  05_run_vanilla_rejection.py --curve [--dry-run]

  05_run_vanilla_rejection.py --emit-tasks --sweep --temperatures 0.6,0.9 [--dry-run]
  05_run_vanilla_rejection.py --time-one 24 --sweep [--dry-run]
  05_run_vanilla_rejection.py --task 0 --sweep [--dry-run]
  05_run_vanilla_rejection.py --submit --sweep [--dry-run]
  05_run_vanilla_rejection.py --curve --temperature 0.6 [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib  # noqa: E402
from lib import phase4                # noqa: E402

RESOURCE_KEY = "phase4_rejection"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tasks", action="store_true",
                       help="Write slurm/tasks_phase4.tsv for both Phase 4 experiments")
    group.add_argument("--task", type=int, help="Run one scaffold's pool by task_id")
    group.add_argument("--submit", action="store_true",
                       help="Submit the rejection tasks as a Slurm array")
    group.add_argument("--time-one", type=int, nargs="?", const=0, metavar="TASK_ID",
                       help="Submit a single scaffold for timing before sizing the array")
    group.add_argument("--curve", action="store_true",
                       help="Build results/rejection_curve.csv from the finished pools")
    parser.add_argument("--array", default=None,
                        help="Override the --array spec, e.g. 3,7 to rerun failures")
    parser.add_argument("--sweep", action="store_true",
                        help="Operate on the temperature sweep: "
                             "slurm/tasks_phase4_tsweep.tsv and the suffixed outputs")
    parser.add_argument("--temperatures", default="0.6,0.9",
                        help="Comma-separated temperatures for --emit-tasks --sweep")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Which temperature's pools --curve should read")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be read, written or submitted")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_tasks:
        if args.sweep:
            temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
            phase4.emit_tsweep_tasks(cfg, temps, args.dry_run)
        else:
            phase4.emit_tasks(cfg, args.dry_run)
    elif args.task is not None:
        phase4.run_rejection(cfg, cluster, args.task, args.dry_run, args.sweep)
    elif args.submit:
        phase4.submit(cluster, args.dry_run, args.array, "rejection", RESOURCE_KEY,
                      args.sweep)
    elif args.time_one is not None:
        phase4.time_one(cluster, args.dry_run, args.time_one, RESOURCE_KEY, args.sweep)
    elif args.curve:
        phase4.build_curve(cfg, args.dry_run, args.temperature)


if __name__ == "__main__":
    main()

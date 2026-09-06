#!/usr/bin/env python3
"""Phase 4b: the random-charge negative control.

For each scaffold and target, mutate randomly chosen designable positions to K
or R (raising the charge) or D or E (lowering it) until the target net charge is
reached. 10 replicates per cell. PLAN.md Section 7. All the work lives in
`lib/phase4.py`; this is the CLI for the control arm.

This isolates what the learned prior contributes. The charge arithmetic and the
designable position set are held identical to the supercharging runs, so the
only thing removed is ProteinMPNN's opinion about which substitution belongs
where. The three points where PLAN.md's one-sentence specification needed a
resolution are documented on `lib/phase4.random_charge_design`.

Threading and energetic scoring of these designs is Phase 5, not this script.

--emit-tasks writes the task file for BOTH Phase 4 experiments, so it only needs
running once whichever script invokes it.

Usage:
  06_run_random_control.py --emit-tasks [--dry-run]
  06_run_random_control.py --task 25 [--dry-run]
  06_run_random_control.py --submit [--array=27,31] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib  # noqa: E402
from lib import phase4                # noqa: E402

RESOURCE_KEY = "phase4_random"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tasks", action="store_true",
                       help="Write slurm/tasks_phase4.tsv for both Phase 4 experiments")
    group.add_argument("--task", type=int,
                       help="Run one scaffold's targets by task_id")
    group.add_argument("--submit", action="store_true",
                       help="Submit the random-control tasks as a Slurm array")
    parser.add_argument("--array", default=None,
                        help="Override the --array spec, e.g. 27,31 to rerun failures")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be read, written or submitted")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_tasks:
        phase4.emit_tasks(cfg, args.dry_run)
    elif args.task is not None:
        phase4.run_random_control(cfg, cluster, args.task, args.dry_run)
    elif args.submit:
        phase4.submit(cluster, args.dry_run, args.array, "random_control", RESOURCE_KEY)


if __name__ == "__main__":
    main()

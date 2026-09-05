#!/usr/bin/env python3
"""Phase 3 baseline: AvNAPSA supercharging.

Runs the AvNAPSA mode of PyRosetta's Supercharge mover using the settings in
`Former_Methods/eGFP/run_pyrosetta_supercharge.py`, which is not modified. All
the work lives in `lib/baseline.py`; this is the CLI for the `avnapsa` arm.

PLAN.md Section 6 offers a reimplementation path "if the Rosetta supercharge
application is not available" and requires validating any reimplementation
against the published AscG-30 and AscG+36 mutation sets. That path is not taken:
PyRosetta exposes the real mover, so this is the genuine AvNAPSA protocol rather
than a reimplementation, the validation requirement does not apply, and the
baseline is labelled `avnapsa`, not `avnapsa_reimpl`.

nstruct is 1, per the frozen design matrix in PLAN.md Section 5, which calls
AvNAPSA deterministic by construction. Note that the reference runs in
`Former_Methods/eGFP/TargetPos2_Avn/` produced 65, 65 and 61 mutations across
three replicates, so that assumption looks questionable. Raising nstruct is a
schema change and goes through PLAN.md first.

--emit-tasks writes the task file for BOTH baseline arms, so it only needs
running once whichever script invokes it.

Usage:
  03_run_avnapsa.py --emit-tasks [--dry-run]
  03_run_avnapsa.py --task 0 [--dry-run]
  03_run_avnapsa.py --submit [--array=17,43,91] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import baseline              # noqa: E402
from lib import config as config_lib  # noqa: E402

METHOD = "avnapsa"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tasks", action="store_true",
                       help="Write slurm/tasks_phase3.tsv for both baseline arms")
    group.add_argument("--task", type=int, help="Run one cell by task_id")
    group.add_argument("--submit", action="store_true",
                       help="Submit this arm's cells as a Slurm array")
    parser.add_argument("--array", default=None,
                        help="Override the --array spec, e.g. 17,43,91 to rerun failures")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be read, written or submitted")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_tasks:
        baseline.emit_tasks(cfg, args.dry_run)
    elif args.task is not None:
        baseline.run_task(cfg, cluster, args.task, METHOD, args.dry_run)
    elif args.submit:
        baseline.submit(cluster, args.dry_run, args.array, METHOD)


if __name__ == "__main__":
    main()

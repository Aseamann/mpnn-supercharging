#!/usr/bin/env python3
"""Phase 3 baseline: Rosetta score-based supercharging.

Runs the score-based mode of PyRosetta's Supercharge mover using the settings in
`Former_Methods/eGFP/run_pyrosetta_supercharge.py`, which is not modified. All
the work lives in `lib/baseline.py`; this is the CLI for the `rosetta` arm.

PLAN.md Section 6 makes this baseline the first thing to cut if the Rosetta
binary is unavailable. It is not cut: PyRosetta exposes
`pyrosetta.rosetta.protocols.design_opt.Supercharge` with every method the
protocol needs, so the real thing runs.

nstruct is 10, per PLAN.md Section 5, specifically to test the claim that
Rosetta is near-deterministic. PyRosetta is initialised once per cell with a
recorded constant seed and the 10 replicates run in sequence off that single RNG
stream, which is how Rosetta's own nstruct behaves and keeps the whole cell
reproducible from one seed. `n_unique_mutation_sets` in the sidecar is the
number the diversity comparison turns on.

Surface definition is `surface_residue_cutoff(16)`, not the
`-surface_atom_cutoff 120` named in PLAN.md Section 6; that parameter governs
AvNAPSA-mode surface definition and does not apply to this mode. See
lib/baseline.py.

--emit-tasks writes the task file for BOTH baseline arms, so it only needs
running once whichever script invokes it.

Usage:
  04_run_rosetta_supercharge.py --emit-tasks [--dry-run]
  04_run_rosetta_supercharge.py --task 200 [--method rosetta] [--dry-run]
  04_run_rosetta_supercharge.py --submit --method rosetta_hbond_off [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import baseline              # noqa: E402
from lib import config as config_lib  # noqa: E402

# The score-based arms. `rosetta` is the reference-faithful one, with
# dont_mutate_hbonded_sidechains left True as run_pyrosetta_supercharge.py sets
# it. `rosetta_hbond_off` sets it False, which is the direct analogue of passing
# -mhbond on the MPNN side, and runs on the same control subset as
# mpnn_soluble_hbond_protected. Which one a task belongs to is the `method`
# column of the task file; slurm/array_baseline.sbatch reads it and passes it
# back in with --method.
METHODS = ("rosetta", "rosetta_hbond_off")
METHOD = METHODS[0]


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
    parser.add_argument("--method", default=METHOD, choices=METHODS,
                        help="Which score-based arm this task belongs to")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be read, written or submitted")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()

    if args.emit_tasks:
        baseline.emit_tasks(cfg, args.dry_run)
    elif args.task is not None:
        baseline.run_task(cfg, cluster, args.task, args.method, args.dry_run)
    elif args.submit:
        baseline.submit(cluster, args.dry_run, args.array, args.method)


if __name__ == "__main__":
    main()

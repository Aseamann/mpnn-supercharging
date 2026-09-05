"""Single source of truth for net-charge math.

Nothing else in the benchmark computes (K+R) - (D+E). If a charge number
appears in a results file, it came through this module.

The definition is deliberately narrow and matches protein_mpnn_supercharge.py's
net_charge() with -addhis absent, which is how every arm of this benchmark runs:
no histidine, no terminal charges, no pKa model.

Run `python charge.py --verify` to check parity against the repo script rather
than taking this docstring's word for it.
"""

from __future__ import annotations

POSITIVE_RESIDUES = ("K", "R")
NEGATIVE_RESIDUES = ("D", "E")
HISTIDINE = "H"


def net_charge(seq: str, add_histidine: bool = False) -> int:
    """Net charge of a sequence: (K+R) - (D+E), optionally +H.

    Mirrors protein_mpnn_supercharge.py:61. Kept as a separate implementation
    rather than an import so the benchmark does not depend on the repo script
    being importable, and verified against it by --verify.
    """
    charge = sum(seq.count(aa) for aa in POSITIVE_RESIDUES)
    charge -= sum(seq.count(aa) for aa in NEGATIVE_RESIDUES)
    if add_histidine:
        charge += seq.count(HISTIDINE)
    return charge


def charge_density(charge: int, n_residues: int) -> float:
    """Net charge per 100 residues."""
    if n_residues <= 0:
        raise ValueError(f"n_residues must be positive, got {n_residues}")
    return 100.0 * charge / n_residues


def delta_q_from_density(delta_q_density: float, n_residues: int) -> int:
    """Convert a size-normalized charge step into an absolute charge step.

    delta_q = round(delta_q_density * n_residues / 100), per PLAN.md Section 5.

    Python's round() is banker's rounding, which would matter at exact .5 ties.
    For the frozen ladder (+/-4, 8, 16, 24) no tie is reachable at any integer
    n_residues, because d*n = 50 (mod 100) has no solution for those d. The
    assertion below is not decorative: it fires if the ladder ever changes to a
    value that can tie, so the rounding rule gets revisited instead of silently
    biasing every affected cell toward the even integer.
    """
    if n_residues <= 0:
        raise ValueError(f"n_residues must be positive, got {n_residues}")
    exact = delta_q_density * n_residues / 100.0
    if abs(exact - int(exact)) == 0.5:
        raise ValueError(
            f"delta_q_density={delta_q_density} at n_residues={n_residues} lands on "
            f"a .5 rounding tie ({exact}). The ladder in benchmark.yaml changed; "
            "fix the rounding rule in PLAN.md Section 5 before continuing."
        )
    return int(round(exact))


def resolve_target(q_wt: int, delta_q_density: float, n_residues: int) -> tuple[int, int]:
    """Return (delta_q, target_charge) for one scaffold at one ladder rung.

    target_charge is the absolute net charge that the repo script's -c expects.
    """
    delta_q = delta_q_from_density(delta_q_density, n_residues)
    return delta_q, q_wt + delta_q


def charge_error(q_actual: int, target_charge: int) -> tuple[int, bool]:
    """Return (absolute error, whether the target was hit exactly)."""
    err = abs(q_actual - target_charge)
    return err, err == 0


def count_mutations(wt_seq: str, design_seq: str) -> tuple[int, list[str]]:
    """Return (n_mutations, ['A12K', ...]) using 1-indexed positions.

    Both sequences must be the same length; a length mismatch means the design
    did not come from this scaffold and is an error, not something to align
    around.
    """
    if len(wt_seq) != len(design_seq):
        raise ValueError(
            f"sequence length mismatch: wt={len(wt_seq)} design={len(design_seq)}"
        )
    muts = [
        f"{wt}{i}{mut}"
        for i, (wt, mut) in enumerate(zip(wt_seq, design_seq), start=1)
        if wt != mut
    ]
    return len(muts), muts


def mut_per_charge(n_mutations: int, q_actual: int, q_wt: int) -> float | None:
    """Mutations per unit of charge actually moved.

    Returns None when the design moved no charge, because dividing by zero here
    would manufacture an infinity that reads as a real measurement downstream.
    """
    achieved = abs(q_actual - q_wt)
    if achieved == 0:
        return None
    return n_mutations / achieved


def _verify_against_repo(repo_root: str) -> int:
    """Check net_charge() parity against protein_mpnn_supercharge.py.

    Imports the repo script's net_charge directly and compares on random
    sequences. Returns a process exit code.
    """
    import importlib.util
    import os
    import random

    script = os.path.join(repo_root, "protein_mpnn_supercharge.py")
    if not os.path.exists(script):
        print(f"FAIL: repo script not found at {script}")
        return 1

    # The repo script imports torch and pyrosetta at module scope, so load only
    # the net_charge function's source rather than executing the whole module.
    src = open(script).read()
    start = src.index("def net_charge(")
    end = src.index("\ndef ", start + 1)
    namespace: dict = {}
    exec(src[start:end], namespace)
    repo_net_charge = namespace["net_charge"]

    rng = random.Random(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    mismatches = 0
    for _ in range(2000):
        seq = "".join(rng.choice(alphabet) for _ in range(rng.randint(80, 250)))
        for add_his in (False, True):
            if net_charge(seq, add_his) != repo_net_charge(seq, add_his):
                mismatches += 1
    if mismatches:
        print(f"FAIL: {mismatches} mismatches against {script}")
        return 1
    print(f"OK: net_charge matches {script}:61 on 2000 random sequences (both -addhis settings)")
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Check net_charge parity against the repo script")
    parser.add_argument("--repo-root", default=None,
                        help="Repo root; defaults to the value in config/cluster.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what --verify would compare, without importing anything")
    args = parser.parse_args()

    if args.repo_root is None:
        # Run directly as a file, so lib/ is on sys.path but its parent is not.
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
        from lib.config import load_cluster  # noqa: E402
        args.repo_root = str(load_cluster()["paths"]["repo_root"])

    if args.dry_run:
        print(f"would compare lib/charge.py net_charge() against "
              f"{args.repo_root}/protein_mpnn_supercharge.py:61")
        print("would test 2000 random sequences, lengths 80 to 250, add_histidine in {False, True}")
        sys.exit(0)
    if args.verify:
        sys.exit(_verify_against_repo(args.repo_root))
    parser.print_help()

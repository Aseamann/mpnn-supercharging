"""Diversity, recovery and mutation parsing.

PLAN.md Section 10.2. These four diversity metrics are what turn the
manuscript's central claim about design diversity from an assertion into a
measurement, so each is defined here once and computed nowhere else.

Every one of them is restricted to the **designable positions**, not the whole
sequence. A method that cannot touch a position cannot contribute diversity
there, and including fixed positions would dilute every score by the same large
constant and compress the differences between arms.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations


def _designable_slice(seq: str, designable: list[int]) -> str:
    """The designable positions of a sequence, 1-based, in ascending order."""
    return "".join(seq[i - 1] for i in sorted(designable) if 1 <= i <= len(seq))


def mutation_set(wt_seq: str, seq: str) -> frozenset[str]:
    """The set of mutations as `A12K` labels. Order-independent by construction."""
    return frozenset(f"{w}{i}{d}" for i, (w, d) in enumerate(zip(wt_seq, seq), 1)
                     if w != d)


def n_unique_mutation_sets(wt_seq: str, seqs: list[str]) -> int:
    """How many distinct mutation sets a cell produced, out of len(seqs).

    Compared as sets, so two designs that reach the same substitutions are one
    result regardless of the order the decoder found them in. AvNAPSA is 1 by
    construction at nstruct 1; the number that carries information is Rosetta's.
    """
    return len({mutation_set(wt_seq, s) for s in seqs})


def mean_pairwise_hamming(seqs: list[str], designable: list[int]) -> float | None:
    """Mean Hamming distance between design pairs, over designable positions.

    Returns None for fewer than two designs, because a single design has no
    pairwise distance and returning 0.0 would read as "no diversity" rather than
    "not measurable". That distinction matters: the AvNAPSA arm has one design
    per cell.
    """
    if len(seqs) < 2:
        return None
    slices = [_designable_slice(s, designable) for s in seqs]
    pairs = list(combinations(slices, 2))
    total = sum(sum(1 for a, b in zip(x, y) if a != b) for x, y in pairs)
    return total / len(pairs)


def positional_entropy(seqs: list[str], designable: list[int]) -> float | None:
    """Mean Shannon entropy, in bits, across designable positions.

    Entropy is computed per position over the designs in the cell, then averaged.
    With n designs the maximum attainable value is log2(n), so this is comparable
    only between cells with the same number of designs; Section 10.2 computes it
    per cell and the comparison is made between arms at equal sample count.
    """
    if len(seqs) < 2:
        return None
    positions = [i for i in sorted(designable) if all(1 <= i <= len(s) for s in seqs)]
    if not positions:
        return None
    total = 0.0
    for i in positions:
        counts = Counter(s[i - 1] for s in seqs)
        n = sum(counts.values())
        total += -sum((c / n) * math.log2(c / n) for c in counts.values())
    return total / len(positions)


def designable_coverage(wt_seq: str, seqs: list[str],
                        designable: list[int]) -> float | None:
    """Union of mutated positions divided by the number of designable positions.

    Measures how much of the available surface the method explored across its
    replicates, as opposed to how different any two designs are.
    """
    if not designable:
        return None
    designable_set = set(designable)
    touched: set[int] = set()
    for seq in seqs:
        touched |= {i for i, (w, d) in enumerate(zip(wt_seq, seq), 1)
                    if w != d and i in designable_set}
    return len(touched) / len(designable_set)


# ---------------------------------------------------------------------------
# Per-design recovery
# ---------------------------------------------------------------------------

def seq_identity(a: str, b: str, positions: list[int] | None = None) -> float:
    """Fraction of identical residues, optionally restricted to 1-based positions."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if positions is None:
        idx = range(1, len(a) + 1)
    else:
        idx = [i for i in positions if 1 <= i <= len(a)]
    idx = list(idx)
    if not idx:
        raise ValueError("no positions to compare")
    return sum(1 for i in idx if a[i - 1] == b[i - 1]) / len(idx)


def frac_designable_mutated(wt_seq: str, seq: str, designable: list[int]) -> float | None:
    """Fraction of this design's designable positions that it actually changed."""
    if not designable:
        return None
    idx = [i for i in designable if 1 <= i <= len(seq)]
    if not idx:
        return None
    return sum(1 for i in idx if wt_seq[i - 1] != seq[i - 1]) / len(idx)


def cell_diversity(wt_seq: str, seqs: list[str], designable: list[int]) -> dict:
    """All four Section 10.2 metrics for one scaffold x method x target cell."""
    return {
        "n_designs": len(seqs),
        "n_unique_mutation_sets": n_unique_mutation_sets(wt_seq, seqs),
        "mean_pairwise_hamming": mean_pairwise_hamming(seqs, designable),
        "positional_entropy": positional_entropy(seqs, designable),
        "designable_coverage": designable_coverage(wt_seq, seqs, designable),
    }

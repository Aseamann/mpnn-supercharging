"""Sequence selection in BLOSUM62 distance space, and MDS for visualisation only.

PLAN.md Section 11.3, which answers Reviewer 1: t-SNE distances are not
meaningful, and the manuscript's selection was performed in that embedded space.
The fix is not to change the embedding, it is to **stop selecting in an embedding
at all**:

* the **medoid** minimises the sum of distances to all other sequences,
* the **divergent representative** maximises its distance from the medoid,

both computed on the original distance matrix. `classical_mds` exists only so a
figure can show the arrangement, and any caption using it has to say so.
Classical MDS (PCoA) is preferred over t-SNE because it preserves distances, so
the objection does not merely move.

**The distance definition here is a reconstruction, not the manuscript's.**
Section 11.3 says to build the BLOSUM62 pairwise score matrix "as before", but no
selection code exists anywhere in this repository, so there is nothing to copy.
The definition used is stated explicitly instead:

    S(a, b)  = sum over aligned positions of BLOSUM62[a_i][b_i]
    d(a, b)  = 1 - S(a, b) / sqrt(S(a, a) * S(b, b))

That is the standard self-score normalisation: it is 0 for identical sequences,
grows with dissimilarity, and is symmetric. It can exceed 1 for sequences whose
cross score is negative, which is a real property of BLOSUM and not clipped away.
Because every comparison here is between equal-length designs threaded on one
scaffold, no alignment step is involved and positions correspond directly.

Whether the corrected procedure changes the eGFP selection is an empirical
question Section 11.3 asks to be reported either way; `compare_selection` returns
what is needed for that sentence.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _blosum62():
    """BLOSUM62 as a dict of (aa, aa) -> score, from Biopython."""
    from Bio.Align import substitution_matrices                    # noqa: PLC0415

    matrix = substitution_matrices.load("BLOSUM62")
    table: dict[tuple[str, str], float] = {}
    for (a, b), score in matrix.items():
        table[(a, b)] = float(score)
        table[(b, a)] = float(score)
    return table


def pair_score(a: str, b: str) -> float:
    """Summed BLOSUM62 score over aligned positions of two equal-length sequences."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    table = _blosum62()
    total = 0.0
    for x, y in zip(a, b):
        if (x, y) in table:
            total += table[(x, y)]
        else:
            # An unknown residue contributes nothing rather than crashing the
            # whole cell. Designs come from a 20-letter alphabet, so this is
            # reachable only through a corrupt FASTA, and it is worth surfacing.
            raise ValueError(f"residue pair {x!r},{y!r} is not in BLOSUM62")
    return total


def distance_matrix(seqs: list[str]) -> list[list[float]]:
    """Symmetric BLOSUM62 distance matrix. See the module docstring for d()."""
    n = len(seqs)
    self_scores = [pair_score(s, s) for s in seqs]
    for i, s in enumerate(self_scores):
        if s <= 0:
            raise ValueError(f"sequence {i} has a non-positive self score ({s}); "
                             "the normalisation is undefined")
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = pair_score(seqs[i], seqs[j])
            d = 1.0 - s / ((self_scores[i] * self_scores[j]) ** 0.5)
            D[i][j] = D[j][i] = d
    return D


def medoid_index(D: list[list[float]]) -> int:
    """Index minimising the summed distance to every other sequence.

    Ties break to the lowest index, so the result is deterministic given the
    input order, and the input order is the sample index.
    """
    if not D:
        raise ValueError("empty distance matrix")
    sums = [sum(row) for row in D]
    best = min(range(len(sums)), key=lambda i: (sums[i], i))
    return best


def divergent_index(D: list[list[float]], medoid: int) -> int:
    """Index furthest from the medoid, ties breaking to the lowest index."""
    if len(D) < 2:
        raise ValueError("need at least two sequences for a divergent representative")
    candidates = [i for i in range(len(D)) if i != medoid]
    return max(candidates, key=lambda i: (D[medoid][i], -i))


def select(seqs: list[str]) -> dict:
    """Medoid and divergent representative, selected in the distance space."""
    D = distance_matrix(seqs)
    med = medoid_index(D)
    out = {"n": len(seqs), "medoid_index": med,
           "medoid_distance_sum": sum(D[med])}
    if len(seqs) >= 2:
        div = divergent_index(D, med)
        out["divergent_index"] = div
        out["divergent_distance_from_medoid"] = D[med][div]
    else:
        out["divergent_index"] = None
        out["divergent_distance_from_medoid"] = None
    return out


def classical_mds(D: list[list[float]], n_components: int = 2):
    """Classical MDS (PCoA). **Visualisation only, never selection.**

    Returns an (n, n_components) array of coordinates and the eigenvalues, so a
    caption can report how much of the distance structure the plot actually
    carries. Negative eigenvalues are possible because BLOSUM-derived distances
    are not guaranteed Euclidean; they are returned rather than hidden.
    """
    import numpy as np                                             # noqa: PLC0415

    d = np.asarray(D, dtype=float)
    n = d.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (d ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    keep = eigvals[:n_components]
    coords = eigvecs[:, :n_components] * np.sqrt(np.maximum(keep, 0))
    return coords, eigvals


def compare_selection(seqs: list[str], previous_indices: list[int] | None = None) -> dict:
    """Corrected selection, plus whether it differs from a previously chosen set.

    Section 11.3 asks explicitly for the answer to "did the selected sequences
    change", including when the answer is no.
    """
    result = select(seqs)
    chosen = [i for i in (result["medoid_index"], result["divergent_index"])
              if i is not None]
    result["selected_indices"] = chosen
    if previous_indices is not None:
        result["previous_indices"] = sorted(previous_indices)
        result["selection_changed"] = sorted(chosen) != sorted(previous_indices)
    return result

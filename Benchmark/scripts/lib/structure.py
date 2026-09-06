"""Structure-prediction parsing: per-residue pLDDT, TM-align, CA RMSD.

PLAN.md Section 9. Not in the Section 3 tree, which lists `lib/metrics.py` for
"diversity, recovery, mutation parsing"; this is a separate concern and is kept
separate rather than overloading that module.

Two Section 9 requirements shape everything here:

* `ca_rmsd_to_wt_crystal` and `tm_score_to_wt_crystal` are measured against the
  **wild-type crystal chain**, never against the threaded model, which already
  assumes the answer.
* `plddt_min_designable` is the worst designable position, not the worst
  position overall, so it catches local misfolding in the region the method
  actually changed while ignoring termini that are floppy in every prediction.
"""

from __future__ import annotations

from pathlib import Path

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _cif_atom_site_columns(path: Path) -> list[str]:
    """The `_atom_site` loop header, in file order.

    mmCIF columns are whitespace-delimited and their order is not fixed: the
    ESMFold2 files put `B_iso_or_equiv` before `Cartn_x`, which no PDB-derived
    ordering does. Every CIF reader here parses this header rather than assuming
    positions.
    """
    columns: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                columns.append(line.strip().split(".", 1)[1])
            elif line.startswith("ATOM") and columns:
                break
    if not columns:
        raise ValueError(f"{path}: no _atom_site loop header found")
    return columns


def read_ca_trace_cif(cif_path: str | Path) -> tuple[list, str]:
    """CA coordinates and sequence from an mmCIF, first chain only.

    Separate from the PDB reader because mmCIF is whitespace-delimited, not
    fixed-column. Handing a CIF to the PDB reader silently yields zero atoms:
    `line[12:16]` lands in the middle of an unrelated field, nothing matches
    "CA", and the caller gets an empty trace that looks like a parse of an empty
    file. That is how the AF3 collection failed on its first run.
    """
    cif_path = Path(cif_path)
    columns = _cif_atom_site_columns(cif_path)
    try:
        i_atom = columns.index("label_atom_id")
        i_comp = columns.index("label_comp_id")
        i_chain = columns.index("label_asym_id")
        i_seq = columns.index("label_seq_id")
        i_x, i_y, i_z = (columns.index(f"Cartn_{a}") for a in "xyz")
    except ValueError as exc:
        raise ValueError(f"{cif_path}: missing _atom_site column: {exc}") from exc

    coords, seq, seen = [], [], set()
    chain_id: str | None = None
    with open(cif_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            fields = line.split()
            if fields[i_atom] != "CA":
                continue
            chain = fields[i_chain]
            if chain_id is None:
                chain_id = chain
            elif chain != chain_id:
                continue
            key = (chain, fields[i_seq])
            if key in seen:
                continue
            seen.add(key)
            seq.append(THREE_TO_ONE.get(fields[i_comp].upper(), "X"))
            coords.append((float(fields[i_x]), float(fields[i_y]),
                           float(fields[i_z])))
    return coords, "".join(seq)


def read_ca_trace_any(path: str | Path) -> tuple[list, str]:
    """CA trace from either a PDB or an mmCIF, dispatched on the suffix."""
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        return read_ca_trace_cif(path)
    return read_ca_trace(path)


def read_ca_trace(pdb_path: str | Path) -> tuple[list[tuple[float, float, float]], str]:
    """CA coordinates and one-letter sequence, in file order, first chain only.

    Deliberately minimal rather than a Biopython structure parse: these files
    are single-chain predictions and single-chain cleaned crystals, and the only
    things needed downstream are the CA trace and the sequence. Altlocs other
    than the first are skipped so a crystal with alternate conformations does
    not contribute two CAs for one residue.
    """
    coords: list[tuple[float, float, float]] = []
    seq: list[str] = []
    seen: set[tuple[str, str]] = set()
    chain_id: str | None = None

    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain = line[21]
            if chain_id is None:
                chain_id = chain
            elif chain != chain_id:
                continue
            key = (chain, line[22:27])
            if key in seen:
                continue
            seen.add(key)
            resname = line[17:20].strip().upper()
            seq.append(THREE_TO_ONE.get(resname, "X"))
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords, "".join(seq)


def read_plddt_per_residue(cif_path: str | Path) -> list[float]:
    """Per-residue pLDDT from the mmCIF `_atom_site.B_iso_or_equiv` column.

    **Read from the CIF, not from the `--also-pdb` conversion.** ESMFold2 writes
    the per-residue confidence into `B_iso_or_equiv` in its mmCIF output, but the
    biotite PDB conversion the tool performs alongside it drops the value: every
    B-factor in the converted PDB is 0.00. Parsing the PDB would silently return
    a column of zeros, and `plddt_min_designable` would be 0.0 for every design
    in the benchmark while looking like a real measurement.

    The column order in these files is not the PDB-derived conventional one
    (`B_iso_or_equiv` precedes `Cartn_x`), so the loop header is parsed for the
    column index rather than positions being assumed.
    """
    columns, values = [], []
    seen: set[tuple[str, str]] = set()
    idx: tuple[int, int, int, int] | None = None
    with open(cif_path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                columns.append(line.strip().split(".", 1)[1])
                continue
            if not line.startswith("ATOM"):
                continue
            if idx is None:
                if not columns:
                    raise ValueError(f"{cif_path}: ATOM records before the loop header")
                try:
                    idx = (columns.index("label_atom_id"),
                           columns.index("B_iso_or_equiv"),
                           columns.index("label_asym_id"),
                           columns.index("label_seq_id"))
                except ValueError as exc:
                    raise ValueError(
                        f"{cif_path}: missing _atom_site column: {exc}") from exc
            i_atom, i_b, i_chain, i_seq = idx
            fields = line.split()
            if fields[i_atom] != "CA":
                continue
            key = (fields[i_chain], fields[i_seq])
            if key in seen:
                continue
            seen.add(key)
            values.append(float(fields[i_b]))
    if not values:
        raise ValueError(f"{cif_path}: no CA records parsed")
    return values


def tm_align(coords_a, seq_a, coords_b, seq_b) -> dict:
    """TM-score and CA RMSD after TM-align superposition.

    `tmtools` 0.3.0, pip-installed into py311 for exactly this. Returns the
    TM-score normalised by the *reference* (chain B, the wild-type crystal),
    which is the convention Section 9 needs: every design is compared against the
    same reference length, so the scores are comparable across designs.

    The RMSD reported is TM-align's, over the aligned pairs after its
    superposition, not an all-atom RMSD over an assumed one-to-one mapping.
    """
    import numpy as np                                             # noqa: PLC0415
    from tmtools import tm_align as _tm_align                      # noqa: PLC0415

    if len(coords_a) < 3 or len(coords_b) < 3:
        raise ValueError(f"too few CA atoms to align: {len(coords_a)} vs {len(coords_b)}")
    res = _tm_align(np.asarray(coords_a, dtype=float),
                    np.asarray(coords_b, dtype=float), seq_a, seq_b)
    return {"tm_score": float(res.tm_norm_chain2),
            "tm_score_norm_query": float(res.tm_norm_chain1),
            "ca_rmsd": float(res.rmsd)}


def plddt_metrics(plddt: list[float], designable: list[int]) -> dict:
    """Mean pLDDT and the worst designable position.

    `designable` is 1-based, matching the LayerSelector cache. Positions outside
    the prediction's length are dropped and counted rather than silently
    indexing off the end; that can only happen if the prediction and the cache
    disagree about the scaffold, which is worth seeing in the record.
    """
    if not plddt:
        raise ValueError("no per-residue pLDDT values")
    in_range = [i for i in designable if 1 <= i <= len(plddt)]
    out = {
        "plddt_mean": sum(plddt) / len(plddt),
        "plddt_min": min(plddt),
        "n_designable_scored": len(in_range),
        "n_designable_out_of_range": len(designable) - len(in_range),
    }
    out["plddt_min_designable"] = min((plddt[i - 1] for i in in_range), default=None)
    out["plddt_mean_designable"] = (
        sum(plddt[i - 1] for i in in_range) / len(in_range) if in_range else None)
    return out

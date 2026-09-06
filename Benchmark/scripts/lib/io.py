"""FASTA header parsing, design identity, seed derivation, schema validation.

The designs.csv column list here is frozen. PLAN.md Section 10.1 is the
authority: change it there first, then here, then in the notebook assertions.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen schema, PLAN.md Section 10.1
# ---------------------------------------------------------------------------

DESIGNS_COLUMNS = [
    "design_id", "scaffold_id", "pdb_id", "fold_class", "n_residues", "q_wt",
    "method", "weights", "hbond_filter", "seed", "sample_index",
    "delta_q_density", "target_charge", "q_actual", "q_error_abs", "hit_exact",
    "n_mutations", "mutations", "mut_per_charge",
    "frac_designable_mutated", "seq_identity_to_wt_designable", "seq_identity_to_wt_global",
    "mpnn_score_masked", "mpnn_score_global",
    "final_temperature", "n_retries", "wall_seconds", "status", "fail_reason",
    "thread_tier", "d_reu_per_res", "d_fa_atr", "d_fa_rep", "d_fa_sol", "d_fa_elec",
    "d_hbond_sc", "d_hbond_bb_sc", "d_hbond_strong", "d_hbond_weak",
    "plddt_mean", "plddt_min_designable", "d_plddt_vs_wt_pred",
    "ca_rmsd_to_wt_crystal", "tm_score_to_wt_crystal",
    "af3_plddt", "af3_ca_rmsd", "af3_msa_mode",
]

MANIFEST_COLUMNS = [
    "scaffold_id", "pdb_id", "chain", "n_residues", "resolution", "fold_class",
    "frac_helix", "frac_strand", "frac_coil", "q_wt", "q_wt_per100",
    "n_designable", "source_split", "is_focus",
]

RUNTIME_COLUMNS = [
    "phase", "cell_id", "slurm_job_id", "slurm_array_task_id",
    "partition", "node", "cpus", "gpus", "wall_seconds", "status",
]

VALID_STATUS = {"ok", "failed"}


# ---------------------------------------------------------------------------
# The one place hbond_filter is derived
# ---------------------------------------------------------------------------

def hbond_filter_from_flag(mutate_hbonded_sidechains: bool) -> bool:
    """Map the -mhbond flag to the frozen hbond_filter column.

    The column reads backwards from the flag name and this is the only place
    that inversion is allowed to happen. PLAN.md Section 5:

      -mhbond passed      -> protection OFF -> hbond_filter = False (primary arm)
      -mhbond absent      -> protection ON  -> hbond_filter = True  (control arm)
    """
    return not mutate_hbonded_sidechains


# ---------------------------------------------------------------------------
# Design identity and seeds
# ---------------------------------------------------------------------------

def control_subset(manifest: list[dict], per_class: int) -> set[str]:
    """The h-bond control subset: the focus scaffold plus `per_class` per class.

    Selection is by sorted scaffold_id within each fold class, so it is a pure
    function of the manifest and reproduces exactly. It lives here rather than
    in one of the phase scripts because two arms in two different phases have to
    land on the identical set: `mpnn_soluble_hbond_protected` in Phase 2 and
    `rosetta_hbond_off` in Phase 3. F9's protection on/off comparison is only
    readable if all four series sit on one scaffold set, and two copies of this
    rule would be free to drift apart.
    """
    chosen = {r["scaffold_id"] for r in manifest
              if str(r["is_focus"]).lower() == "true"}
    by_class: dict[str, list[dict]] = {}
    for row in manifest:
        if row["fold_class"]:
            by_class.setdefault(row["fold_class"], []).append(row)
    for members in by_class.values():
        members.sort(key=lambda r: r["scaffold_id"])
        chosen.update(r["scaffold_id"] for r in members[:per_class])
    return chosen


def cell_id(scaffold_id: str, method: str, target_charge: int) -> str:
    """Identifier for one (scaffold, method, target) cell of the design matrix."""
    return f"{scaffold_id}_{method}_q{target_charge:+d}"


def design_id(scaffold_id: str, method: str, target_charge: int, sample_index: int) -> str:
    """Identifier for one design. Unique across the whole benchmark."""
    return f"{cell_id(scaffold_id, method, target_charge)}_s{sample_index:02d}"


def cell_seed(base_seed: int, scaffold_id: str, method: str, target_charge: int) -> int:
    """Derive the seed passed to np.random.seed before calling the repo script.

    Decision A. protein_mpnn_supercharge.py:642 draws its own seed from an
    unseeded RNG over 0..999 and never records it. Seeding numpy immediately
    beforehand makes that draw deterministic, so the value recorded here fully
    determines the run even though it is not the integer the script uses
    internally.

    One seed per cell, not per sample: a cell is a single invocation with
    -n 10, so all ten designs share the seed that produced them and are told
    apart by sample_index. Rerunning the cell with this seed regenerates all
    ten. Per-sample seeds would mean ten separate invocations and ten model
    loads for no gain in reproducibility.

    Derived rather than sequential so that rerunning one failed cell reproduces
    its original seed without needing its position in the task file. Reduced to
    32 bits because np.random.seed rejects anything wider.
    """
    key = f"{base_seed}|{scaffold_id}|{method}|{target_charge}"
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32 - 1)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

@dataclass
class FastaRecord:
    name: str
    charge: int
    score: float | None          # absent on Phase 3 baseline records
    global_score: float | None   # absent on Phase 3 baseline records
    temperature: float | None    # absent on the native and baseline records
    sequence: str

    @property
    def is_native(self) -> bool:
        """Whether this is the wild-type record rather than a sampled design.

        Keyed off the name, not off `temperature`. The MPNN arms omit
        temperature only on the native record, but the Phase 3 baselines omit it
        everywhere, so a temperature test would call every baseline record
        native and split_native would then reject the file. Sampled designs are
        named `<scaffold_id>_<index>`; scaffold ids end in a chain letter
        (`1a1x_A`) or are plain (`eGFP`), so a trailing `_<digits>` is
        unambiguous.
        """
        return re.search(r"_\d+$", self.name) is None


def _parse_header(header: str) -> dict:
    """Parse a comma-separated key=value FASTA header from the repo script.

    Two shapes are emitted (protein_mpnn_supercharge.py:889 and :893):
      >NAME,charge=I,score=F,global_score=F                  (native)
      >NAME_I,charge=I,score=F,global_score=F,temperature=F  (design)
    """
    header = header.lstrip(">").strip()
    parts = header.split(",")
    fields: dict = {"name": parts[0]}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"malformed FASTA header field {part!r} in {header!r}")
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_fasta(path: str | Path) -> list[FastaRecord]:
    """Parse a design FASTA into records, native first.

    Raises on any header that does not parse. Phase 2 acceptance requires every
    header to parse cleanly, so a silent skip here would hide a real failure.
    """
    path = Path(path)
    records: list[FastaRecord] = []
    name: str | None = None
    fields: dict = {}
    chunks: list[str] = []

    def flush() -> None:
        if name is None:
            return
        # charge is always required. score and global_score are MPNN-specific:
        # the Phase 3 Supercharge mover produces neither, and lib/baseline.py
        # writes charge-only headers rather than inventing numbers to fill them.
        if "charge" not in fields:
            raise ValueError(f"{path}: header {name!r} has no charge")
        score = fields.get("score")
        global_score = fields.get("global_score")
        temp = fields.get("temperature")
        records.append(FastaRecord(
            name=name,
            charge=int(fields["charge"]),
            score=float(score) if score is not None else None,
            global_score=float(global_score) if global_score is not None else None,
            temperature=float(temp) if temp is not None else None,
            sequence="".join(chunks),
        ))

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                fields = _parse_header(line)
                name = fields["name"]
                chunks = []
            else:
                chunks.append(line)
        flush()

    if not records:
        raise ValueError(f"{path}: no records parsed")
    return records


def split_native(records: list[FastaRecord]) -> tuple[FastaRecord, list[FastaRecord]]:
    """Separate the native record from the sampled designs."""
    natives = [r for r in records if r.is_native]
    designs = [r for r in records if not r.is_native]
    if len(natives) != 1:
        raise ValueError(f"expected exactly 1 native record, found {len(natives)}")
    return natives[0], designs


# ---------------------------------------------------------------------------
# LayerSelector cache -> ProteinMPNN fixed-positions JSONL
# ---------------------------------------------------------------------------

def load_designable_cache(pkl_path: str | Path) -> tuple[str, list[int], int]:
    """Read the Phase 1 LayerSelector cache written at
    protein_mpnn_supercharge.py:695.

    The pickle is a 3-tuple (wt_seq, designable_indices, q_wt). The indices are
    1-based positions into wt_seq and already exclude Gly, Pro and Cys, because
    the cache was built with mutate_glyprocys False. Anything reading this cache
    inherits that protection and must not reapply it.
    """
    import pickle                                                  # noqa: PLC0415

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    wt_seq, indices, q_wt = data[:3]
    indices = [int(i) for i in indices]
    out_of_range = [i for i in indices if i < 1 or i > len(wt_seq)]
    if out_of_range:
        raise ValueError(
            f"{pkl_path}: designable indices outside 1..{len(wt_seq)}: "
            f"{out_of_range[:10]}. This is the PDB-versus-pose numbering bug; the "
            "scaffold was not renumbered to 1..N before the cache was built."
        )
    return wt_seq, indices, int(q_wt)


def fixed_positions_for(name: str, chain_id: str, wt_seq: str,
                        designable: list[int]) -> dict:
    """Build the fixed-positions dict protein_mpnn_run.py expects.

    This is the exact construction protein_mpnn_supercharge.py:703-706 performs
    before calling tied_featurize:

        for i in range(1, len(wt_seq) + 1):
            if i not in indices:
                fixed_positions_dict[pdb_file][chain_id].append(i)

    Reproducing it here rather than approximating it is the whole point of the
    Phase 4 ablation: the vanilla sampler has to be restricted to exactly the
    positions the supercharging run was free to change, or the comparison is
    between two different design problems.

    `name` must match the key ProteinMPNN derives for the structure, which is
    the PDB basename without its extension (protein_mpnn_utils.py:190).
    Positions are 1-based; tied_featurize zeroes `mask[np.array(fixed) - 1]`.
    """
    designable_set = set(designable)
    fixed = [i for i in range(1, len(wt_seq) + 1) if i not in designable_set]
    return {name: {chain_id: fixed}}


def write_fixed_positions_jsonl(path: str | Path, mapping: dict) -> Path:
    """Write a fixed-positions dict as the single-line JSONL the runner reads.

    protein_mpnn_run.py:80-85 iterates the file's lines and keeps the last JSON
    object, so one object on one line is the format that behaves predictably.
    """
    import json                                                    # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(json.dumps(mapping) + "\n")
    return path


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

RETRY_MARKER = "Increasing temperature to: "
TEMP_EXCEEDED_MARKER = "Temperature too high for"


def count_retries(log_path: str | Path) -> int:
    """Count temperature escalations in a per-cell stdout log.

    n_retries is in no output file. protein_mpnn_supercharge.py:462 prints one
    line per escalation and that print is unconditional, so the count is exact
    as long as the log was retained. Retries are part of the true cost of the
    method and are not discarded.
    """
    with open(log_path) as fh:
        return sum(1 for line in fh if RETRY_MARKER in line)


def temperature_exceeded(log_path: str | Path) -> bool:
    """Whether the run gave up after passing the 0.9 temperature ceiling."""
    with open(log_path) as fh:
        return any(TEMP_EXCEEDED_MARKER in line for line in fh)


# ---------------------------------------------------------------------------
# CSV writing and validation
# ---------------------------------------------------------------------------

def validate_row(row: dict, columns: list[str] = DESIGNS_COLUMNS) -> dict:
    """Check a row against the frozen schema before it reaches a CSV.

    Unknown keys are an error rather than a warning: they mean the writer and
    PLAN.md Section 10.1 have drifted apart.
    """
    unknown = set(row) - set(columns)
    if unknown:
        raise ValueError(f"columns not in the frozen schema: {sorted(unknown)}")
    if row.get("status") not in VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(VALID_STATUS)}, got {row.get('status')!r}")
    if row["status"] == "failed" and not row.get("fail_reason"):
        raise ValueError(f"{row.get('design_id')}: status=failed requires a fail_reason")
    return {col: row.get(col, "") for col in columns}


def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    """Write a CSV with the frozen column order, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

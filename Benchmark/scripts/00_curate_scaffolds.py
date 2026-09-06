#!/usr/bin/env python3
"""Phase 1: scaffold curation.

Produces data/scaffold_manifest.csv: 32 scaffolds, 8 per fold class, drawn from
the CATH 4.2 test split used by the ProteinMPNN paper, plus eGFP as scaffold 33.

Stages run in order and each one writes an inspectable intermediate, so a stage
can be rerun without repeating the ones before it:

  pool     data/splits/ + excluded_PDBs.csv -> data/candidates.csv
  fetch    candidates                       -> data/raw/*.pdb
  filter   raw structures                   -> data/filtered.csv
  cluster  filtered sequences               -> data/clustered.csv   (needs mmseqs)
  dssp     cluster representatives          -> data/dssp.csv        (needs mkdssp)
  select   dssp + charge spread             -> data/scaffold_manifest.csv
  targets  manifest + charge ladder         -> config/benchmark.yaml

n_designable is deliberately left empty by `select`. It comes from the
LayerSelector pass in 01_prepare_structures.py, which fills it in afterwards.
Writing a guess there would violate the rule against uncomputed numbers.

Every stage is a no-op under --dry-run and reports what it would read and write.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import charge as charge_lib   # noqa: E402
from lib import config as config_lib   # noqa: E402
from lib import io as bio              # noqa: E402  (lib/io.py, not stdlib io)

STAGES = ["pool", "fetch", "filter", "cluster", "dssp", "select", "targets"]

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

NUCLEIC_RESIDUES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def say_would(msg: str) -> None:
    print(f"[dry-run] would {msg}", flush=True)


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun the '{produced_by}' stage first, or pass --stage all."
        )
    return path


# ---------------------------------------------------------------------------
# stage: pool
# ---------------------------------------------------------------------------

def stage_pool(cfg: dict, cluster: dict, dry_run: bool) -> None:
    """CATH 4.2 test split minus the soluble-weights exclusion list.

    Decision C: the split is the candidate pool and excluded_PDBs.csv is applied
    on top of it, because mpnn_soluble is the primary arm.
    """
    src = cfg["scaffold_source"]
    split_file = config_lib.bench_path(src["split_file"])
    excluded = Path(cluster["paths"]["proteinmpnn"]) / src["excluded_pdbs"]
    out = config_lib.bench_path("data", "candidates.csv")

    if dry_run:
        say_would(f"read split {src['split_key']!r} from {split_file}")
        if not split_file.exists():
            say_would(f"first download it from {src['split_url']} (NOT PRESENT on disk)")
        say_would(f"drop any PDB listed in {excluded} "
                  f"({'present' if excluded.exists() else 'NOT PRESENT'})")
        say_would(f"write {out}")
        return

    if not split_file.exists():
        split_file.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading split from {src['split_url']}")
        urllib.request.urlretrieve(src["split_url"], split_file)

    with open(split_file) as fh:
        splits = json.load(fh)
    if src["split_key"] not in splits:
        raise SystemExit(
            f"{split_file} has no {src['split_key']!r} key (found {sorted(splits)}).\n"
            "This is not the expected artifact. Confirm the authoritative split "
            "before continuing; do not substitute another list."
        )
    entries = splits[src["split_key"]]
    log(f"split {src['split_key']!r}: {len(entries)} chains")

    if not excluded.exists():
        raise SystemExit(f"missing exclusion list {excluded}")

    # This file is a pandas dump: the first column is an unnamed row index and
    # the PDB IDs live under the PDB_IDS header. Reading positionally would
    # silently build an exclusion set of integers that matches nothing, so the
    # column is looked up by name and its absence is an error.
    with open(excluded) as fh:
        reader = csv.DictReader(fh)
        if "PDB_IDS" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{excluded} has no PDB_IDS column (found {reader.fieldnames}). "
                "Confirm the exclusion list format before continuing."
            )
        excluded_ids = {r["PDB_IDS"].strip().lower() for r in reader if r.get("PDB_IDS")}
    log(f"exclusion list: {len(excluded_ids)} PDB IDs from {excluded.name}")

    rows = []
    for entry in entries:
        # Entries look like "1a2b.A" or "1a2bA".
        entry = str(entry).strip()
        if "." in entry:
            pdb_id, chain = entry.split(".", 1)
        else:
            pdb_id, chain = entry[:4], entry[4:] or "A"
        pdb_id = pdb_id.lower()
        if pdb_id in excluded_ids:
            continue
        rows.append({"pdb_id": pdb_id, "chain": chain, "source_split": src["split_name"]})

    log(f"candidates after exclusion: {len(rows)}")
    bio.write_csv(out, rows, ["pdb_id", "chain", "source_split"])
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# stage: fetch
# ---------------------------------------------------------------------------

def stage_fetch(cfg: dict, cluster: dict, dry_run: bool, limit: int | None) -> None:
    candidates_path = config_lib.bench_path("data", "candidates.csv")
    raw_dir = config_lib.bench_path("data", "raw")

    if dry_run:
        if candidates_path.exists():
            n = len(read_csv(candidates_path))
            say_would(f"read {n} candidates from {candidates_path}")
        else:
            say_would(f"read {candidates_path} (NOT PRESENT; the pool stage writes it, "
                      "so the count is not known yet)")
        say_would(f"download the ones missing from {raw_dir} via "
                  f"{RCSB_PDB_URL.format(pdb_id='XXXX')}")
        if limit:
            say_would(f"stop after {limit} structures (--limit)")
        return

    candidates = require(candidates_path, "pool")
    rows = read_csv(candidates)
    if limit:
        rows = rows[:limit]
    todo = [r for r in rows if not (raw_dir / f"{r['pdb_id']}.pdb").exists()]

    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, row in enumerate(todo, 1):
        dest = raw_dir / f"{row['pdb_id']}.pdb"
        try:
            urllib.request.urlretrieve(RCSB_PDB_URL.format(pdb_id=row["pdb_id"].upper()), dest)
        except Exception as exc:                      # noqa: BLE001
            failed.append((row["pdb_id"], str(exc)))
            continue
        if i % 100 == 0:
            log(f"  fetched {i}/{len(todo)}")
    log(f"fetched {len(todo) - len(failed)} structures, {len(failed)} failed")
    if failed:
        fail_path = config_lib.bench_path("logs", "fetch_failures.csv")
        bio.write_csv(fail_path, [{"pdb_id": p, "reason": r} for p, r in failed],
                      ["pdb_id", "reason"])
        log(f"failures recorded in {fail_path}")


# ---------------------------------------------------------------------------
# stage: filter
# ---------------------------------------------------------------------------

def parse_structure(path: Path, chain_id: str) -> dict:
    """Pull the fields the Section 4 filters need out of one PDB file."""
    info = {
        "method": None, "resolution": None, "n_chains": 0,
        "sequence": "", "residue_numbers": [], "ca_coords": [],
        "has_nucleic": False, "missing_backbone": False,
    }
    seen_chains = set()
    backbone_seen: dict[int, set] = {}

    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            # Multi-model entries (NMR, and the occasional X-ray file) repeat
            # every chain once per model. Without this, the residue count is the
            # count times the number of models.
            if rec.startswith("ENDMDL"):
                break
            if rec == "EXPDTA":
                info["method"] = line[10:].strip()
            elif rec == "REMARK" and line[7:10].strip() == "2" and "RESOLUTION." in line:
                # Format: "REMARK   2 RESOLUTION.    1.74 ANGSTROMS."
                # Scanning the whole line for the first parseable float would
                # return the remark number 2, so only the text after
                # "RESOLUTION." is considered. NMR entries say "NOT APPLICABLE"
                # and correctly leave resolution as None.
                tail = line.split("RESOLUTION.", 1)[1]
                for token in tail.split():
                    try:
                        info["resolution"] = float(token)
                        break
                    except ValueError:
                        continue
            elif rec in ("ATOM  ", "HETATM"):
                chain = line[21]
                resname = line[17:20].strip()
                if rec == "ATOM  ":
                    seen_chains.add(chain)
                if resname in NUCLEIC_RESIDUES and rec == "ATOM  ":
                    info["has_nucleic"] = True
                if chain != chain_id or rec != "ATOM  ":
                    continue
                altloc = line[16]
                if altloc not in (" ", "A"):
                    continue
                atom = line[12:16].strip()
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                backbone_seen.setdefault(resseq, set()).add(atom)
                if atom == "CA" and resname in THREE_TO_ONE:
                    info["residue_numbers"].append(resseq)
                    info["sequence"] += THREE_TO_ONE[resname]
                    info["ca_coords"].append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )

    info["n_chains"] = len(seen_chains)
    for resseq in info["residue_numbers"]:
        if not {"N", "CA", "C"} <= backbone_seen.get(resseq, set()):
            info["missing_backbone"] = True
            break
    return info


def max_chain_break(residue_numbers: list[int], ca_coords: list[tuple]) -> int:
    """Longest run of missing residues, by numbering gap and by CA-CA distance."""
    worst = 0
    for i in range(1, len(residue_numbers)):
        gap = residue_numbers[i] - residue_numbers[i - 1] - 1
        if gap > 0:
            worst = max(worst, gap)
        else:
            a, b = ca_coords[i - 1], ca_coords[i]
            dist = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            if dist > 4.5:                      # a break the numbering does not show
                worst = max(worst, 1)
    return worst


def stage_filter(cfg: dict, cluster: dict, dry_run: bool) -> None:
    candidates_path = config_lib.bench_path("data", "candidates.csv")
    raw_dir = config_lib.bench_path("data", "raw")
    out = config_lib.bench_path("data", "filtered.csv")
    f = cfg["filters"]

    if dry_run:
        say_would(f"apply Section 4 filters 2 to 4 to structures in {raw_dir} "
                  f"({'present' if raw_dir.exists() else 'NOT PRESENT'})")
        say_would(f"keep X-ray, resolution <= {f['max_resolution']}, "
                  f"{'single-chain ASU' if f.get('require_single_chain_asu', True) else 'single designed chain'}, "
                  f"{f['min_residues']} to {f['max_residues']} residues, "
                  f"chain breaks <= {f['max_chain_break']}, no nucleic acids")
        say_would(f"compute q_wt through lib/charge.py and write {out}")
        say_would(f"record every rejection and its reason in "
                  f"{config_lib.bench_path('logs', 'filter_rejections.csv')}")
        return

    candidates = require(candidates_path, "pool")
    rows, rejected = [], []
    for row in read_csv(candidates):
        path = raw_dir / f"{row['pdb_id']}.pdb"
        if not path.exists():
            rejected.append((row["pdb_id"], "not fetched"))
            continue
        info = parse_structure(path, row["chain"])

        reason = None
        if not info["method"] or f["experimental_method"] not in info["method"].upper():
            reason = f"method={info['method']}"
        elif info["resolution"] is None or info["resolution"] > f["max_resolution"]:
            reason = f"resolution={info['resolution']}"
        elif f.get("require_single_chain_asu", True) and info["n_chains"] != 1:
            reason = f"n_chains={info['n_chains']}"
        elif not (f["min_residues"] <= len(info["sequence"]) <= f["max_residues"]):
            reason = f"n_residues={len(info['sequence'])}"
        elif info["has_nucleic"]:
            reason = "nucleic acid present"
        elif info["missing_backbone"]:
            reason = "missing backbone atoms"
        else:
            brk = max_chain_break(info["residue_numbers"], info["ca_coords"])
            if brk > f["max_chain_break"]:
                reason = f"chain_break={brk}"

        if reason:
            rejected.append((row["pdb_id"], reason))
            continue

        q_wt = charge_lib.net_charge(info["sequence"], cfg["charge_definition"]["add_histidine"])
        rows.append({
            "pdb_id": row["pdb_id"], "chain": row["chain"],
            "source_split": row["source_split"],
            "n_residues": len(info["sequence"]),
            "resolution": info["resolution"],
            # Recorded even when the strict ASU filter is off, so scaffolds
            # taken from oligomeric entries stay identifiable downstream.
            "n_chains_in_asu": info["n_chains"],
            "q_wt": q_wt,
            "q_wt_per100": round(charge_lib.charge_density(q_wt, len(info["sequence"])), 3),
            "sequence": info["sequence"],
        })

    log(f"passed filters: {len(rows)}, rejected: {len(rejected)}")
    bio.write_csv(out, rows, ["pdb_id", "chain", "source_split", "n_residues",
                              "resolution", "n_chains_in_asu", "q_wt",
                              "q_wt_per100", "sequence"])
    bio.write_csv(config_lib.bench_path("logs", "filter_rejections.csv"),
                  [{"pdb_id": p, "reason": r} for p, r in rejected], ["pdb_id", "reason"])
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# stage: cluster
# ---------------------------------------------------------------------------

def stage_cluster(cfg: dict, cluster: dict, dry_run: bool) -> None:
    filtered = config_lib.bench_path("data", "filtered.csv")
    fasta = config_lib.bench_path("data", "filtered.fasta")
    out = config_lib.bench_path("data", "clustered.csv")
    work = config_lib.bench_path("data", "mmseqs")
    m = cfg["filters"]["mmseqs"]
    mmseqs_bin = Path(cluster["conda"]["mmseqs_env"]) / "bin" / "mmseqs"

    if dry_run:
        say_would(f"write {fasta} from {filtered}")
        say_would(f"run {mmseqs_bin} easy-cluster --min-seq-id {m['min_seq_id']} "
                  f"-c {m['coverage']} in {work}")
        say_would(f"keep one representative per cluster and write {out}")
        if not mmseqs_bin.exists():
            say_would(f"NOTE: {mmseqs_bin} is NOT PRESENT; the real run would stop here")
        return

    require(filtered, "filter")
    if not mmseqs_bin.exists():
        raise SystemExit(f"mmseqs not found at {mmseqs_bin}. Check cluster.yaml conda.mmseqs_env.")

    rows = read_csv(filtered)
    with open(fasta, "w") as fh:
        for row in rows:
            fh.write(f">{row['pdb_id']}_{row['chain']}\n{row['sequence']}\n")

    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "clu"
    cmd = [str(mmseqs_bin), "easy-cluster", str(fasta), str(prefix), str(work / "tmp"),
           "--min-seq-id", str(m["min_seq_id"]), "-c", str(m["coverage"])]
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    reps = set()
    with open(f"{prefix}_cluster.tsv") as fh:
        for line in fh:
            rep, _member = line.split("\t")[:2]
            reps.add(rep.strip())

    kept = [r for r in rows if f"{r['pdb_id']}_{r['chain']}" in reps]
    log(f"clusters at {m['min_seq_id']} identity: {len(reps)}; representatives kept: {len(kept)}")
    bio.write_csv(out, kept, list(kept[0].keys()) if kept else ["pdb_id"])
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# stage: dssp
# ---------------------------------------------------------------------------

def assign_fold_class(fracs: dict, cfg: dict) -> str | None:
    """First match in the frozen precedence order wins; no match means discard.

    Decision F. The Section 4 cutoffs overlap (helix 0.20 / strand 0.20 /
    coil 0.55 satisfies both alpha_beta and loop_rich) and leave gaps
    (helix 0.35 / strand 0.12 / coil 0.53 satisfies none), so neither the order
    nor the discard rule is inferable from the cutoffs alone.
    """
    defs = cfg["fold_classes"]["definitions"]
    for name in cfg["fold_classes"]["precedence"]:
        d = defs[name]
        ok = True
        if "min_helix" in d and fracs["helix"] < d["min_helix"]:
            ok = False
        if "max_helix" in d and fracs["helix"] > d["max_helix"]:
            ok = False
        if "min_strand" in d and fracs["strand"] < d["min_strand"]:
            ok = False
        if "max_strand" in d and fracs["strand"] > d["max_strand"]:
            ok = False
        if "min_coil" in d and fracs["coil"] < d["min_coil"]:
            ok = False
        if ok:
            return name
    return None


def stage_dssp(cfg: dict, cluster: dict, dry_run: bool) -> None:
    clustered = config_lib.bench_path("data", "clustered.csv")
    raw_dir = config_lib.bench_path("data", "raw")
    out = config_lib.bench_path("data", "dssp.csv")
    dssp_bin = Path(cluster["paths"].get("dssp_bin")
                    or Path(cluster["conda"]["benchmark_env"]) / "bin" / "mkdssp")

    if dry_run:
        say_would(f"run {dssp_bin} on each representative in {clustered}")
        say_would("map DSSP 8-state to 3-state as "
                  f"{ {k: v for k, v in cfg['fold_classes']['dssp_map'].items()} }")
        say_would("assign fold class by precedence "
                  f"{cfg['fold_classes']['precedence']}, discarding scaffolds matching none")
        say_would(f"write {out}")
        if not dssp_bin.exists():
            say_would(f"NOTE: {dssp_bin} is NOT PRESENT; install DSSP into py311 first "
                      "(PLAN.md Section 2.1)")
        return

    require(clustered, "cluster")
    if not dssp_bin.exists():
        raise SystemExit(
            f"mkdssp not found at {dssp_bin}. PLAN.md Section 2.1 requires DSSP "
            "installed into py311 before Phase 1. Record the install in "
            "results/environment.json and logs/ISSUES.md."
        )

    from Bio.PDB import DSSP, PDBParser                       # noqa: PLC0415

    dssp_map = cfg["fold_classes"]["dssp_map"]
    parser = PDBParser(QUIET=True)
    rows, discarded = [], []

    for row in read_csv(clustered):
        path = raw_dir / f"{row['pdb_id']}.pdb"
        model = parser.get_structure(row["pdb_id"], path)[0]
        dssp = DSSP(model, str(path), dssp=str(dssp_bin))
        states = [dssp[key][2] for key in dssp.keys() if key[0] == row["chain"]]
        if not states:
            discarded.append((row["pdb_id"], "DSSP returned no states for the chain"))
            continue
        counts = {"helix": 0, "strand": 0, "coil": 0}
        for s in states:
            counts[dssp_map.get(s, "coil")] += 1
        total = len(states)
        fracs = {k: v / total for k, v in counts.items()}
        fold_class = assign_fold_class(fracs, cfg)
        if fold_class is None:
            discarded.append((row["pdb_id"],
                              f"matched no fold class (h={fracs['helix']:.2f} "
                              f"s={fracs['strand']:.2f} c={fracs['coil']:.2f})"))
            continue
        rows.append({**row,
                     "frac_helix": round(fracs["helix"], 4),
                     "frac_strand": round(fracs["strand"], 4),
                     "frac_coil": round(fracs["coil"], 4),
                     "fold_class": fold_class})

    log(f"classified: {len(rows)}, discarded for matching no class: {len(discarded)}")
    for name in cfg["fold_classes"]["precedence"]:
        log(f"  {name}: {sum(1 for r in rows if r['fold_class'] == name)}")
    bio.write_csv(out, rows, list(rows[0].keys()) if rows else ["pdb_id"])
    bio.write_csv(config_lib.bench_path("logs", "fold_class_discards.csv"),
                  [{"pdb_id": p, "reason": r} for p, r in discarded], ["pdb_id", "reason"])
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# stage: select
# ---------------------------------------------------------------------------

def pick_for_class(members: list[dict], n: int, min_positive: int) -> list[dict]:
    """Take n scaffolds spanning the length range, forcing positive-q_wt coverage.

    Section 4 filter 7 wants a spread over length and over WT net charge, with
    at least two positive-q_wt scaffolds per class, otherwise the positive
    supercharging arm is trivially easy everywhere.
    """
    positives = sorted((m for m in members if int(m["q_wt"]) > 0),
                       key=lambda m: int(m["n_residues"]))
    chosen = []
    # Take the positive-charge quota first, spread across the length range.
    if positives:
        want = min(min_positive, len(positives), n)
        step = max(1, len(positives) // want)
        chosen = positives[::step][:want]

    remaining = [m for m in members if m not in chosen]
    remaining.sort(key=lambda m: int(m["n_residues"]))
    slots = n - len(chosen)
    if slots > 0 and remaining:
        step = max(1, len(remaining) // slots)
        chosen += remaining[::step][:slots]
    return sorted(chosen, key=lambda m: int(m["n_residues"]))[:n]


def stage_select(cfg: dict, cluster: dict, dry_run: bool) -> None:
    dssp_csv = config_lib.bench_path("data", "dssp.csv")
    out = config_lib.bench_path("data", "scaffold_manifest.csv")
    sel = cfg["selection"]
    focus = sel["focus_scaffold"]
    egfp_pdb = Path(cluster["paths"]["egfp_pdb"])

    if dry_run:
        say_would(f"read {dssp_csv} and take {sel['n_per_class']} scaffolds per fold class")
        say_would(f"span the length range and require >= {sel['min_positive_q_wt_per_class']} "
                  "scaffolds with positive WT net charge per class")
        say_would(f"add {focus['scaffold_id']} from {egfp_pdb} as scaffold 33 with is_focus=True "
                  f"({'present' if egfp_pdb.exists() else 'NOT PRESENT'})")
        say_would(f"write {out} with columns {bio.MANIFEST_COLUMNS}")
        say_would("leave n_designable empty; 01_prepare_structures.py fills it from "
                  "the LayerSelector pass")
        return

    require(dssp_csv, "dssp")
    by_class: dict[str, list[dict]] = {}
    for row in read_csv(dssp_csv):
        by_class.setdefault(row["fold_class"], []).append(row)

    manifest = []
    for name in cfg["fold_classes"]["precedence"]:
        members = by_class.get(name, [])
        picked = pick_for_class(members, sel["n_per_class"], sel["min_positive_q_wt_per_class"])
        n_pos = sum(1 for m in picked if int(m["q_wt"]) > 0)
        log(f"{name}: {len(members)} available, {len(picked)} selected, {n_pos} with q_wt > 0")
        if len(picked) < sel["n_per_class"]:
            log(f"  WARNING: {name} is short by {sel['n_per_class'] - len(picked)}; "
                "recorded as is, not backfilled from another class")
        if n_pos < sel["min_positive_q_wt_per_class"]:
            log(f"  WARNING: {name} has only {n_pos} positive-q_wt scaffolds, "
                f"below the required {sel['min_positive_q_wt_per_class']}")
        for m in picked:
            manifest.append({
                "scaffold_id": f"{m['pdb_id']}_{m['chain']}",
                "pdb_id": m["pdb_id"], "chain": m["chain"],
                "n_residues": m["n_residues"], "resolution": m["resolution"],
                "fold_class": m["fold_class"],
                "frac_helix": m["frac_helix"], "frac_strand": m["frac_strand"],
                "frac_coil": m["frac_coil"],
                "q_wt": m["q_wt"], "q_wt_per100": m["q_wt_per100"],
                "n_designable": "", "source_split": m["source_split"], "is_focus": False,
            })

    # eGFP, scaffold 33, outside the split.
    if not egfp_pdb.exists():
        raise SystemExit(f"focus scaffold not found at {egfp_pdb}")
    info = parse_structure(egfp_pdb, focus["chain"])
    q_wt = charge_lib.net_charge(info["sequence"], cfg["charge_definition"]["add_histidine"])
    manifest.append({
        "scaffold_id": focus["scaffold_id"], "pdb_id": focus["scaffold_id"],
        "chain": focus["chain"], "n_residues": len(info["sequence"]),
        "resolution": info["resolution"] if info["resolution"] is not None else "",
        "fold_class": "", "frac_helix": "", "frac_strand": "", "frac_coil": "",
        "q_wt": q_wt,
        "q_wt_per100": round(charge_lib.charge_density(q_wt, len(info["sequence"])), 3),
        "n_designable": "", "source_split": focus["source_split"], "is_focus": True,
    })
    log(f"{focus['scaffold_id']}: {len(info['sequence'])} residues, q_wt={q_wt} "
        "(fold class left empty; the dssp stage does not run on the focus scaffold)")

    bio.write_csv(out, manifest, bio.MANIFEST_COLUMNS)
    log(f"wrote {out} with {len(manifest)} rows")


# ---------------------------------------------------------------------------
# stage: targets
# ---------------------------------------------------------------------------

def stage_targets(cfg: dict, cluster: dict, dry_run: bool) -> None:
    manifest_path = config_lib.bench_path("data", "scaffold_manifest.csv")
    ladder = cfg["charge_ladder"]["delta_q_density"]

    if dry_run:
        say_would(f"read {manifest_path} "
                  f"({'present' if manifest_path.exists() else 'NOT PRESENT'})")
        say_would(f"resolve the ladder {ladder} per scaffold as "
                  "delta_q = round(density * n_residues / 100), "
                  "target_charge = q_wt + delta_q")
        say_would("replace the generated resolved_targets block in config/benchmark.yaml, "
                  "preserving the hand-written section above it")
        return

    require(manifest_path, "select")
    targets: dict[str, list[dict]] = {}
    for row in read_csv(manifest_path):
        n_res, q_wt = int(row["n_residues"]), int(row["q_wt"])
        cells = []
        for density in ladder:
            delta_q, target = charge_lib.resolve_target(q_wt, density, n_res)
            cells.append({"delta_q_density": density, "delta_q": delta_q,
                          "target_charge": target})
        targets[row["scaffold_id"]] = cells

    config_lib.write_resolved_targets(targets)
    log(f"wrote resolved targets for {len(targets)} scaffolds "
        f"({len(targets) * len(ladder)} cells) into config/benchmark.yaml")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what each stage would read and write, and do nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of structures fetched, for a quick check")
    args = parser.parse_args()

    cfg = config_lib.load_benchmark()
    cluster = config_lib.load_cluster()
    stages = STAGES if args.stage == "all" else [args.stage]

    for name in stages:
        log(f"\n=== stage: {name} ===")
        if name == "pool":
            stage_pool(cfg, cluster, args.dry_run)
        elif name == "fetch":
            stage_fetch(cfg, cluster, args.dry_run, args.limit)
        elif name == "filter":
            stage_filter(cfg, cluster, args.dry_run)
        elif name == "cluster":
            stage_cluster(cfg, cluster, args.dry_run)
        elif name == "dssp":
            stage_dssp(cfg, cluster, args.dry_run)
        elif name == "select":
            stage_select(cfg, cluster, args.dry_run)
        elif name == "targets":
            stage_targets(cfg, cluster, args.dry_run)


if __name__ == "__main__":
    main()

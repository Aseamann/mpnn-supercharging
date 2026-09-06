"""Config loading. Every script reads paths and partitions through here.

cluster.yaml is the only file a new user must edit, so nothing else in the
benchmark is allowed to hardcode a cluster path.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]

# Marks the machine-generated tail of benchmark.yaml. Everything above the
# begin marker is hand-written and is preserved verbatim on rewrite.
GENERATED_BEGIN = "# ---- BEGIN GENERATED: resolved_targets (written by 00_curate_scaffolds.py) ----"
GENERATED_END = "# ---- END GENERATED ----"


def load_cluster(path: os.PathLike | str | None = None) -> dict:
    """Load cluster.yaml."""
    path = Path(path) if path else CONFIG_DIR / "cluster.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_benchmark(path: os.PathLike | str | None = None) -> dict:
    """Load benchmark.yaml, including any generated resolved_targets block."""
    path = Path(path) if path else CONFIG_DIR / "benchmark.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def bench_path(*parts: str) -> Path:
    """Resolve a path relative to Benchmark/."""
    return BENCHMARK_ROOT.joinpath(*parts)


def write_resolved_targets(targets: dict, path: os.PathLike | str | None = None,
                           dry_run: bool = False) -> str:
    """Replace the generated resolved_targets block in benchmark.yaml.

    PLAN.md Section 5 requires the resolved per-scaffold target list to live in
    config/benchmark.yaml so it is inspectable and frozen before any run starts.
    A yaml round-trip would strip the hand-written comments that carry the
    frozen decisions, so the generated block is delimited and replaced as text.

    Returns the text that was written (or would be written under dry_run).
    """
    path = Path(path) if path else CONFIG_DIR / "benchmark.yaml"
    current = open(path).read()

    if GENERATED_BEGIN in current:
        head = current.split(GENERATED_BEGIN)[0]
    else:
        head = current.rstrip() + "\n\n"

    block = yaml.safe_dump({"resolved_targets": targets}, sort_keys=True,
                           default_flow_style=False)
    new_text = f"{head}{GENERATED_BEGIN}\n{block}{GENERATED_END}\n"

    if not dry_run:
        with open(path, "w") as fh:
            fh.write(new_text)
    return new_text


def exclude_flag(cluster: dict) -> list[str]:
    """sbatch --exclude arguments for the nodes cluster.yaml blacklists.

    Returns an empty list when nothing is excluded, so callers can splice it into
    a command unconditionally. Keeping the node list in cluster.yaml means a bad
    node is one edit in the one file a new user is expected to touch, rather than
    a flag remembered at each submission.
    """
    nodes = str(cluster.get("slurm", {}).get("exclude_nodes", "") or "").strip()
    return [f"--exclude={nodes}"] if nodes else []


def arm_by_name(benchmark: dict, name: str) -> dict:
    for arm in benchmark["arms"]:
        if arm["name"] == name:
            return arm
    raise KeyError(f"no arm named {name!r} in benchmark.yaml")


def primary_arm(benchmark: dict) -> dict:
    for arm in benchmark["arms"]:
        if arm.get("primary"):
            return arm
    raise KeyError("no arm marked primary in benchmark.yaml")

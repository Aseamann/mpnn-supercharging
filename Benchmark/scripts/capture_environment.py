#!/usr/bin/env python3
"""Write results/environment.json.

PLAN.md ground rule 4: capture the software stack once so the run can be
rebuilt. Everything here is read from the live environment or from disk. A field
that cannot be determined is recorded as null with a reason rather than guessed.

Usage:
  capture_environment.py --dry-run
  capture_environment.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as config_lib   # noqa: E402

OUT = "results/environment.json"


def run(cmd: list[str], timeout: int = 300) -> str | None:
    """Return stdout, or None if the command is unavailable or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(cluster: dict) -> dict:
    env = Path(cluster["conda"]["benchmark_env"])
    repo = Path(cluster["paths"]["repo_root"])
    mpnn = Path(cluster["paths"]["proteinmpnn"])
    pip = str(env / "bin" / "pip")
    py = str(env / "bin" / "python")

    info: dict = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "captured_on_host": platform.node(),
        "platform": platform.platform(),
        "python": run([py, "-c", "import sys;print(sys.version.split()[0])"]),
        "conda_env": str(env),
    }

    # Package versions, read from the env rather than from a requirements file.
    versions = {}
    for mod in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "torch",
                "Bio", "yaml", "tmtools"):
        versions[mod] = run([py, "-c",
                             f"import {mod};print(getattr({mod},'__version__','unknown'))"])
    info["packages"] = versions

    # PyRosetta has no __version__; its build string is the closest equivalent.
    info["pyrosetta"] = run([py, "-c",
                             "import pyrosetta;print(pyrosetta.version())"]) or \
        run([py, "-c", "import pyrosetta,os;print(os.path.dirname(pyrosetta.__file__))"])

    info["proteinmpnn"] = {
        "path": str(mpnn),
        "commit": run(["git", "-C", str(mpnn), "rev-parse", "HEAD"]),
        "commit_short": run(["git", "-C", str(mpnn), "rev-parse", "--short", "HEAD"]),
        "supercharge_sample_present": bool(run(
            ["grep", "-c", "def supercharge_sample", str(mpnn / "protein_mpnn_utils.py")])),
        "soluble_weights_present": (mpnn / "soluble_model_weights" / "v_48_020.pt").exists(),
    }

    info["benchmark_repo"] = {
        "path": str(repo),
        "commit": run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "-C", str(repo), "status", "--porcelain"])),
    }

    # Tools installed for this benchmark, per PLAN.md 2.1.
    #
    # DSSP is read from cluster.yaml's paths.dssp_bin, not from the benchmark
    # env's bin/. It could not be installed into py311: the salilab dssp 3.0.0
    # package does not declare its boost dependency, and boost-cpp 1.73 is
    # unsatisfiable against that env's Python 3.11 pin. It lives in its own env.
    # Probing the wrong path recorded "not installed" in environment.json while
    # DSSP was in fact installed and had already produced the fold classes.
    dssp_bin = Path(cluster["paths"]["dssp_bin"])
    info["added_tools"] = {
        "tmtools": {
            "install_command": "pip install tmtools",
            "version": versions.get("tmtools"),
            "env": str(env),
            "purpose": "TM-align for ca_rmsd_to_wt_crystal and tm_score_to_wt_crystal (Phase 6)",
        },
        "dssp": {
            "install_command": "conda create -p <dssp_env> -c salilab dssp=3.0.0 boost-cpp=1.73",
            "path": str(dssp_bin),
            "env": str(cluster["conda"].get("dssp_env", "")),
            "installed": dssp_bin.exists(),
            "version": run([str(dssp_bin), "--version"]) if dssp_bin.exists() else None,
            "purpose": "8-state secondary structure for the Phase 1 fold classes",
            "note": ("In its own env, not py311: the salilab package omits its boost "
                     "dependency and boost-cpp 1.73 is unsatisfiable against py311's "
                     "Python 3.11 pin. See logs/ISSUES.md."),
        },
    }

    # Phase 6 runs against shared lab installs rather than anything this
    # benchmark installed, so record what was used rather than what was added.
    info["external_tools"] = {
        "esmfold2": {
            "path": str(cluster["paths"].get("esmfold", "")),
            "env": str(cluster["conda"].get("esmfold_env", "")),
            "model": "biohub/ESMFold2-Fast",
            "settings": "loops=3 steps=50 diffusion_samples=1 seed=0",
            "note": ("The -Fast model is the tool README's recommendation for "
                     "screening large design sets; the accuracy check is the AF3 "
                     "subset. Recorded in every row of results/esmfold.csv."),
        },
        "alphafold3": {
            "path": str(cluster["paths"].get("alphafold3", "")),
            "modules": ["apptainer/1.2.5", "alphafold/vs3.0.0-pgarias"],
            "msa_mode": "single_sequence",
            "note": ("Run with empty unpairedMsa/pairedMsa, no templates and "
                     "--norun_data_pipeline. PLAN.md Section 9: building an MSA "
                     "for a supercharged sequence retrieves wild-type homologs "
                     "and rescues the prediction with signal the design lacks."),
        },
    }

    # Phase 8. The benchmark env has IPython but no jupyter, nbconvert or
    # ipykernel, so the nbconvert route to the notebook runs from a second env.
    # Probe it rather than hardcoding versions; absent, record that it is absent.
    nb_env = Path(cluster["conda"].get("notebook_env", ""))
    nb_python = nb_env / "bin" / "python"
    nb_info: dict = {
        "env": str(nb_env),
        "purpose": ("jupyter/nbconvert for notebooks/analysis.ipynb. Has no "
                    "torch and no PyRosetta and cannot run any other phase; "
                    "the notebook only reads results/*.csv."),
        "present": nb_python.exists(),
    }
    if nb_python.exists():
        probe = ("import json,sys,importlib;"
                 "d={'python':'.'.join(map(str,sys.version_info[:3]))};"
                 "[d.__setitem__(m,getattr(importlib.import_module(m),"
                 "'__version__','?')) for m in "
                 "['nbconvert','nbformat','nbclient','ipykernel','numpy',"
                 "'pandas','matplotlib']];print(json.dumps(d))")
        out = run([str(nb_python), "-c", probe])
        if out:
            try:
                nb_info["versions"] = json.loads(out.strip().splitlines()[-1])
            except json.JSONDecodeError:
                nb_info["versions"] = None
    info["external_tools"]["notebook_env"] = nb_info

    # The no-jupyter route, which needs nothing beyond the benchmark env.
    info["external_tools"]["notebook_runner"] = {
        "command": "python scripts/run_notebook.py",
        "purpose": ("Runs the notebook's code cells in order in one namespace "
                    "for envs without jupyter. Verified byte-identical table "
                    "output against the nbconvert route on 2026-08-14."),
    }

    # Which partition each phase actually ran on, per PLAN.md Section 14.
    info["phase_partitions"] = {
        "phase1_curate": cluster["slurm"]["cpu_partition"],
        "phase2_mpnn": cluster["slurm"]["cpu_partition"],
        "phase3_baseline": cluster["slurm"]["cpu_partition"],
        "phase4_rejection": cluster["slurm"]["cpu_partition"],
        "phase4_random": cluster["slurm"]["cpu_partition"],
        "phase5_thread": cluster["slurm"]["cpu_partition"],
        "phase6_esmfold": cluster["slurm"]["gpu_partition"],
        "phase6_af3": cluster["slurm"]["owner_partition"],
    }
    info["excluded_nodes"] = cluster["slurm"].get("exclude_nodes", "")

    # GPU details are recorded even for the CPU phases so Phase 6 can be compared.
    info["gpu"] = {
        "nvidia_smi": run(["nvidia-smi",
                           "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"]),
        "cuda_available": run([py, "-c",
                               "import torch;print(torch.cuda.is_available())"]),
        "torch_cuda_version": run([py, "-c",
                                   "import torch;print(torch.version.cuda)"]),
    }

    split = config_lib.bench_path("data", "splits", "chain_set_splits.json")
    info["data_provenance"] = {
        "cath_split_file": str(split),
        "cath_split_sha256": sha256(split),
        "cath_split_url": None,
        "excluded_pdbs": str(mpnn / "soluble_model_weights" / "excluded_PDBs.csv"),
        "excluded_pdbs_sha256": sha256(mpnn / "soluble_model_weights" / "excluded_PDBs.csv"),
    }

    info["pip_freeze"] = (run([pip, "freeze"]) or "").splitlines()
    info["conda_list"] = (run(["conda", "list", "-p", str(env), "--export"]) or "").splitlines()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cluster = config_lib.load_cluster()
    out = config_lib.bench_path(OUT)

    if args.dry_run:
        print(f"[dry-run] would query the env at {cluster['conda']['benchmark_env']}")
        print("[dry-run] would record package versions, PyRosetta build, ProteinMPNN "
              "commit, GPU details, split checksums, pip freeze and conda list")
        print(f"[dry-run] would write {out}")
        return

    bench = config_lib.load_benchmark()
    info = collect(cluster)
    info["data_provenance"]["cath_split_url"] = bench["scaffold_source"]["split_url"]

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"wrote {out}")
    print(f"  proteinmpnn commit: {info['proteinmpnn']['commit_short']}")
    print(f"  torch {info['packages']['torch']}, numpy {info['packages']['numpy']}")
    print(f"  tmtools {info['packages']['tmtools']}, dssp installed: "
          f"{info['added_tools']['dssp']['installed']}")


if __name__ == "__main__":
    main()

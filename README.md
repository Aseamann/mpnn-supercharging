# mpnn-supercharging

> **Note:** Publication rights to this work are held exclusively by Austin Seamann. A paper describing this method is currently in preparation — please reach out before citing or building on this work.

A toolkit for protein surface supercharging using [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and [PyRosetta](https://www.pyrosetta.org/). Given one or more PDB structures, the main script samples sequences redesigned to hit a target net charge (or a range of charges) while preserving structure-critical residues. Designed sequences can optionally be threaded back onto the backbone and relaxed with PyRosetta FastRelax.

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
  - [1. Clone this repository](#1-clone-this-repository)
  - [2. Clone ProteinMPNN and install the modified utils](#2-clone-proteinmpnn-and-install-the-modified-utils)
  - [3. Set up a Python environment](#3-set-up-a-python-environment)
  - [4. Install PyRosetta](#4-install-pyrosetta)
- [Scripts](#scripts)
  - [protein\_mpnn\_supercharge.py](#protein_mpnn_superchargepy)
  - [threading\_only.py](#threading_onlypy)
  - [run\_supercharge.sh](#run_superchargesh)
- [Usage Examples](#usage-examples)
- [Output](#output)
- [Design Notes](#design-notes)

---

## Overview

Supercharging is a protein engineering strategy that increases the net surface charge of a protein to improve solubility, reduce aggregation, or modulate binding properties. This toolkit:

1. **Detects surface residues** using PyRosetta's `LayerSelector`.
2. **Filters non-mutable positions**: residues forming strong sidechain hydrogen bonds, residues adjacent to a user-defined catalytic site, and (optionally) glycine, proline, and cysteine.
3. **Runs ProteinMPNN** with a custom supercharging sampler (`supercharge_sample`) that iteratively mutates only mutable surface residues until the target net charge is achieved.
4. **Outputs a FASTA file** of designed sequences with per-sequence scores and charges.
5. Optionally **threads each designed sequence** onto the WT backbone using PyRosetta FastRelax, producing relaxed PDB structures.

A standalone threading utility (`threading_only.py`) is also provided for cases where you already have a FASTA of designed sequences and only want to build structures.

---

## Dependencies

| Package | Notes |
|---|---|
| [PyRosetta](https://www.pyrosetta.org/) | **Optional** — only required for surface residue detection (when no cache exists) and for `--thread`. Requires a free academic license; install via wheel from the PyRosetta download page. |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Cloned from GitHub; path passed via `--path_to_weights` |
| Python ≥ 3.8 | |
| PyTorch | ProteinMPNN runs well on CPU; a GPU will speed up large batches but is not required |
| NumPy | |
| pandas | |

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/aseamann/mpnn-supercharging.git
cd mpnn-supercharging
```

### 2. Clone ProteinMPNN and install the modified utils

This toolkit requires a custom `supercharge_sample` method that is **not** present in the standard ProteinMPNN release. You must replace the standard `protein_mpnn_utils.py` in your ProteinMPNN clone with the modified version provided here.

```bash
# Clone the standard ProteinMPNN repository
git clone https://github.com/dauparas/ProteinMPNN.git /path/to/ProteinMPNN

# Option A — Replace the whole utils file (recommended):
cp Modified_ProteinMPNN/protein_mpnn_utils.py /path/to/ProteinMPNN/protein_mpnn_utils.py

# Option B — Patch only the new method:
# Open both files and copy the `supercharge_sample` method from
# Modified_ProteinMPNN/protein_mpnn_utils.py into the ProteinMPNN class
# in /path/to/ProteinMPNN/protein_mpnn_utils.py.
```

The `Modified_ProteinMPNN/protein_mpnn_utils_ori.py` file is the unmodified original for reference.

At runtime, pass `--path_to_weights /path/to/ProteinMPNN` so the script can locate both the model weights (in `vanilla_model_weights/` or `soluble_model_weights/`) and the modified `protein_mpnn_utils.py`.

### 3. Set up a Python environment

Start from the same conda environment used by ProteinMPNN, then add the one extra dependency needed by this toolkit:

```bash
# Create the environment following the ProteinMPNN recommendation
conda create --name mlfold
conda activate mlfold

# Install PyTorch — visit https://pytorch.org/get-started/locally/ for the
# command matching your CUDA version. Example for CUDA 11.3:
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# NumPy is pulled in with PyTorch; add the one extra dependency:
conda install pandas
```

### 4. Install PyRosetta (optional — only needed for surface detection and threading)

PyRosetta is required for two steps:
- **Surface residue detection**: run when no `parsed/{pdb_id}_seq_indices.pkl` cache exists.
- **Sequence threading** (`--thread` flag): build relaxed 3D structures from designed sequences.

If you only want to run supercharging on a cached protein or skip threading, PyRosetta is not needed. When a code path that requires it is reached without PyRosetta installed, the script will print a clear error and exit.

Download the appropriate wheel for your platform from [https://www.pyrosetta.org/downloads](https://www.pyrosetta.org/downloads), then install:

```bash
conda activate mlfold
pip install pyrosetta-*.whl
```

No additional installation is required for this repo; the scripts are run directly.

---

## Scripts

### `protein_mpnn_supercharge.py`

Main script. Samples supercharged sequences for one or more PDB structures using ProteinMPNN's learned protein language model.

#### How it works

1. Loads each PDB with PyRosetta and identifies surface residues via `LayerSelector`.
2. Optionally removes residues near a catalytic site (within `--distance` Å) and residues forming strong sidechain H-bonds (identified by a short FastRelax run).
3. Fixes all non-surface residues so ProteinMPNN only redesigns the mutable surface.
4. Samples sequences with `supercharge_sample`, which biases amino acid selection toward charged residues to meet the target net charge at a given temperature.
5. If the target charge is not reached, temperature is gradually increased (up to 0.9) before giving up (controlled by `--unrestrict`).
6. Writes results to a FASTA file and (optionally) threads each sequence onto the WT backbone.

#### Arguments

**I/O**

| Flag | Default | Description |
|---|---|---|
| `-i` / `--input` | *(required)* | Directory of `.pdb` files to supercharge |
| `-o` / `--output` | `{pdb}_sc_{charge}.fasta` | Output FASTA filename |
| `-n` / `--num_samples` | `1` | Number of sequences to sample per PDB |
| `--chain_id` | `A` | Chain to redesign |

**Preservation**

| Flag | Default | Description |
|---|---|---|
| `-cat` / `--catalytic` | — | Comma-separated PDB residue numbers to protect (e.g. `"100,150,200"`) |
| `-d` / `--distance` | — | Neighbourhood radius (Å) around catalytic residues to additionally protect; set to `0.0` to protect only the catalytic residues themselves |
| `-f` / `--fixed_positions_jsonl` | — | JSONL file of pre-defined fixed positions (ProteinMPNN format) |
| `-gpc` / `--mutate_glyprocys` | `False` | Allow mutation of Gly, Pro, Cys (off by default) |
| `-mhbond` / `--mutate_hbonded_sidechains` | `False` | Allow mutation of surface residues with strong sidechain H-bonds (off by default) |
| `-nofast` / `--no_fastrelax` | `False` | Skip FastRelax when detecting H-bonds (faster but less accurate; only relevant when `-mhbond` is **not** set) |

**Model**

| Flag | Default | Description |
|---|---|---|
| `--model` | `v_48_020` | ProteinMPNN model checkpoint: `v_48_002`, `v_48_010`, `v_48_020`, `v_48_030`, or `ALL` |
| `--weights` | `original` | Weight set: `original` (vanilla) or `soluble` |
| `--path_to_weights` | `/home/als515/GitHub_Repos/ProteinMPNN` | Absolute path to your ProteinMPNN clone |

**Supercharging**

| Flag | Default | Description |
|---|---|---|
| `-c` / `--target_charge` | `0` | Target net charge for the designed sequence |
| `-top` / `--top_charge` | — | Upper bound for a charge scan (used together with `--bottom_charge`) |
| `-bottom` / `--bottom_charge` | — | Lower bound for a charge scan |
| `-p` / `--probability_threshold` | `0.01` | Minimum ProteinMPNN probability to allow at any position |
| `-t` / `--temperature` | `0.3` | Sampling temperature (lower = more conservative) |
| `-u` / `--unrestrict` | `False` | Gradually raise temperature if the target charge is not achieved in one pass |
| `-addhis` / `--add_histidine` | `False` | Count His as +1 charge (in addition to Lys and Arg) |

**Threading**

| Flag | Default | Description |
|---|---|---|
| `--thread` | `False` | Thread designed sequences onto the WT backbone after sampling |
| `--thread_dir` | `threaded` | Output directory for threaded PDB structures |
| `--thread_workers` | `8` | Parallel worker processes for threading (each spawns its own PyRosetta instance) |

**General**

| Flag | Default | Description |
|---|---|---|
| `-v` / `--verbose` | `False` | Print per-run energies, mutation lists, full sequences, and debug info |

---

### `threading_only.py`

Standalone utility: given an existing FASTA of designed sequences and a backbone PDB, thread each sequence onto the structure using PyRosetta FastRelax and produce relaxed PDB files.

This is useful when you have already run `protein_mpnn_supercharge.py` (or any other design pipeline) and want to build 3D models without re-running ProteinMPNN.

#### Arguments

| Flag | Default | Description |
|---|---|---|
| `-i` / `--input` | *(required)* | Input backbone PDB file |
| `-f` / `--fasta` | *(required)* | FASTA file of designed sequences |
| `-o` / `--output_dir` | `threaded` | Output directory for threaded PDB files |
| `--num_relax` | `5` | Number of independent FastRelax runs per sequence; lowest-energy structure is kept |
| `--workers` | `8` | Parallel worker processes (each spawns its own PyRosetta) |
| `--skip_wt` | `False` | Skip sequences identical to the WT |
| `-v` / `--verbose` | `False` | Print per-run energies and mutation details |

---

### `run_supercharge.sh`

SLURM job submission script for running `protein_mpnn_supercharge.py` on a compute cluster.

Usage:
```bash
sbatch run_supercharge.sh <target_charge>
# Example: design sequences with net charge +10
sbatch run_supercharge.sh 10
```

The script assumes:
- Input PDBs are in a directory named `pdb/` in the current working directory.
- The cluster has modules for `gcc/14.2.0`.
- GPU is available (requests 1 GPU via `--gres=gpu:1`).

Edit the `#SBATCH` header lines and the `python` command at the bottom to match your cluster environment and desired flags.

---

## Usage Examples

### eGFP example (included)

A ready-to-run example using eGFP is provided in `examples/eGFP/`. The PDB structure is already in place. After completing installation, run:

```bash
python protein_mpnn_supercharge.py \
    -i examples/eGFP/pdb \
    -o examples/eGFP/eGFP_sc10.fa \
    -c 10 \
    -n 5 \
    -t 0.3 \
    --model v_48_020 \
    --path_to_weights /path/to/ProteinMPNN \
    -mhbond \
    -u \
    -v
```

Expected output: 5 designed sequences at net charge +10 (starting from WT charge −7), written to `examples/eGFP/eGFP_sc10.fa`. Surface-residue indices are cached to `parsed/eGFP_seq_indices.pkl` after the first run. See [examples/eGFP/README.md](examples/eGFP/README.md) for full walkthrough details.

---

### Supercharge to a single target charge

```bash
python protein_mpnn_supercharge.py \
    -i pdb/ \
    -o my_protein_sc10.fa \
    -c 10 \
    -n 20 \
    -t 0.3 \
    --model v_48_020 \
    --weights original \
    --path_to_weights /path/to/ProteinMPNN \
    -mhbond \
    -u \
    -v
```

### Scan a range of charges

```bash
python protein_mpnn_supercharge.py \
    -i pdb/ \
    -o charge_scan.fa \
    -bottom -5 \
    -top 15 \
    -n 5 \
    -t 0.3 \
    --path_to_weights /path/to/ProteinMPNN
```

### Protect a catalytic triad (residues 57, 102, 195) and their 8 Å neighbourhood

```bash
python protein_mpnn_supercharge.py \
    -i pdb/ \
    -o protected_sc10.fa \
    -c 10 \
    -cat "57,102,195" \
    -d 8.0 \
    -n 10 \
    --path_to_weights /path/to/ProteinMPNN
```

### Supercharge and immediately thread sequences onto the backbone

```bash
python protein_mpnn_supercharge.py \
    -i pdb/ \
    -o sc10.fa \
    -c 10 \
    -n 5 \
    --path_to_weights /path/to/ProteinMPNN \
    --thread \
    --thread_dir threaded_sc10 \
    --thread_workers 4
```

### Thread an existing FASTA independently

```bash
python threading_only.py \
    -i my_protein.pdb \
    -f sc10.fa \
    -o threaded/ \
    --num_relax 5 \
    --workers 4 \
    -v
```

---

## Output

### FASTA file

Each sampled sequence is written as a FASTA entry:
```
>pdb_id_0,charge=+10,score=1.2345,global_score=1.1234,temperature=0.3
MKVLSLAEGKVKEEAEKAEEQAEKDAEEKAE...
```

- `charge`: net charge of the designed sequence (K+R−D−E, optionally +H)
- `score`: ProteinMPNN negative log-likelihood over redesigned positions only
- `global_score`: ProteinMPNN negative log-likelihood over the full sequence
- `temperature`: sampling temperature actually used

The wild-type sequence is always written as the first entry for reference.

### Threaded PDB files

When `--thread` is used (or `threading_only.py` is run), each designed sequence is threaded onto the WT backbone via PyRosetta FastRelax. The pipeline:
1. Identifies mutated residues relative to WT.
2. Builds a constrained FastRelax that forces only the mutated positions and repacks a shell of neighbouring residues.
3. Runs `--num_relax` independent relaxations and keeps the lowest-energy structure.
4. Writes the relaxed structure to `{thread_dir}/{seq_id}.pdb`.

A summary table is printed after threading completes, including sequence length, charge, Rosetta energy, and mutation list.

### Parsed cache

On first run, `protein_mpnn_supercharge.py` saves surface-residue and net-charge information to `parsed/{pdb_id}_seq_indices.pkl`. Subsequent runs with the same PDB re-use this cache, skipping the PyRosetta surface-detection step. Delete the `parsed/` directory to force recomputation.

---

## Design Notes

- **Temperature**: Lower temperatures (0.1–0.3) produce sequences closer to the WT; higher temperatures (0.5–0.9) explore more sequence space. Use `--unrestrict` to let the script raise the temperature automatically if the target charge cannot be achieved.
- **H-bond protection**: By default, surface residues that form strong sidechain hydrogen bonds (energy < −0.5 REU) are not-excluded from redesign. Use `-mhbond` to enable this filter. A short FastRelax is performed before H-bond evaluation; add `-nofast` to skip it if you have already relaxed the structure with PyRosetta.
- **Catalytic site protection**: Use `-cat` with a comma-separated list of PDB residue numbers to lock a catalytic site. The `--distance` argument extends the exclusion zone to all residues within the given radius.
- **Gly/Pro/Cys**: These residues are excluded by default because mutations can disrupt backbone conformation or disulfide bonds. Enable with `-gpc`.
- **Parallelism**: Threading is parallelised using Python `multiprocessing` with the `spawn` start method (required because PyRosetta cannot be shared across forked processes). Set `--thread_workers` to match your available CPU cores.

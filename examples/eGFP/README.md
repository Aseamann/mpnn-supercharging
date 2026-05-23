# eGFP Supercharging Example

This directory contains a ready-to-run example of MPNN-based protein supercharging applied to enhanced GFP (eGFP, PDB: 1EMA). The input PDB is in `pdb/eGFP.pdb`.

---

## What this example does

- Detects the surface-exposed residues of eGFP using PyRosetta's `LayerSelector`.
- Excludes Gly, Pro, Cys, and residues forming strong sidechain H-bonds from redesign.
- Runs ProteinMPNN's `supercharge_sample` to iteratively mutate surface residues until the target net charge is reached.
- Outputs a FASTA file with the WT sequence and each designed sequence annotated with charge, score, and sampling temperature.

eGFP has a WT net charge of **−7**. This example targets **+10**, requiring ~13–15 surface substitutions per sequence.

---

## Prerequisites

Before running this example, complete the full installation described in the [main README](../../README.md). In particular:

### 1. Install ProteinMPNN with the modified utils

```bash
# Clone the standard ProteinMPNN repository
git clone https://github.com/dauparas/ProteinMPNN.git /path/to/ProteinMPNN

# Copy the modified protein_mpnn_utils.py (contains supercharge_sample method)
cp ../../Modified_ProteinMPNN/protein_mpnn_utils.py /path/to/ProteinMPNN/protein_mpnn_utils.py
```

The script uses `sys.path.append(args.path_to_weights)` at runtime to import `protein_mpnn_utils` from your ProteinMPNN directory, so the modified file must be in place before running.

If you prefer a minimal patch rather than replacing the whole file, copy the `supercharge_sample` method (from `Modified_ProteinMPNN/protein_mpnn_utils.py`) into the `ProteinMPNN` class in the standard `protein_mpnn_utils.py`.

### 2. Set up a Python environment

```bash
# Create the environment following the ProteinMPNN recommendation
conda create --name mlfold
conda activate mlfold

# Install PyTorch — see https://pytorch.org/get-started/locally/ for your CUDA version
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Add the one extra dependency
conda install pandas
```

### 3. Install PyRosetta (optional — only needed for this example)

For this example, PyRosetta is required to detect surface residues (no pre-built cache is included). It is also required if you add `--thread`.

Download the wheel for your platform from [https://www.pyrosetta.org/downloads](https://www.pyrosetta.org/downloads), then:

```bash
conda activate mlfold
pip install pyrosetta-*.whl
```

If you skip PyRosetta and run without a cache, the script will exit with a clear error message directing you here.

---

## Running the example

Run from the **repository root** (`mpnn-supercharging/`):

```bash
python protein_mpnn_supercharge.py \
    -i examples/eGFP/pdb \
    -o examples/eGFP/eGFP_sc10.fa \
    -c 10 \
    -n 5 \
    -t 0.3 \
    --model v_48_020 \
    --weights original \
    --path_to_weights /path/to/ProteinMPNN \
    -mhbond \
    -u \
    -v
```

### Flag explanation

| Flag | Value | Reason |
|------|-------|--------|
| `-i examples/eGFP/pdb` | — | Input directory containing `eGFP.pdb` |
| `-o examples/eGFP/eGFP_sc10.fa` | — | Output FASTA path |
| `-c 10` | +10 | Target net charge |
| `-n 5` | 5 | Number of sequences to sample |
| `-t 0.3` | 0.3 | Sampling temperature (conservative) |
| `--model v_48_020` | — | ProteinMPNN checkpoint (48-neighbour, noise σ=0.20) |
| `--weights original` | — | Standard vanilla model weights |
| `--path_to_weights` | *your path* | Root of your ProteinMPNN clone |
| `-mhbond` | — | Allow mutation of H-bonded surface residues |
| `-u` | — | Raise temperature if target charge not reached in one pass |
| `-v` | — | Verbose: print per-mutation updates and final summary table |

---

## Expected output

### Console (with `-v`)

```
Surface indices: [1, 3, 4, 6, ...]
Adjusted indices: [1, 3, 6, 9, ...]     ← after removing Gly/Pro/Cys
Initial net charge: {'eGFP': -7}
Final indices: [1, 6, 9, 11, ...]       ← mutable positions
Updating residue D196 in batch 0 with R likelihood 0.09 ...
Current sequence for batch 0 at charge -5: VSKGEEL...
...
  [ 0] charge=+10  score=1.54  global_score=1.04  temp=0.3
  [ 1] charge=+10  score=1.45  global_score=1.00  temp=0.3
  ...
```

### Output FASTA (`eGFP_sc10.fa`)

```
>eGFP,charge=-7,score=1.3218,global_score=0.9558
VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK...   ← WT

>eGFP_0,charge=10,score=1.5432,global_score=1.0384,temperature=0.3
VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDARYGKLRLK...   ← design 0

>eGFP_1,charge=10,score=1.4470,global_score=0.9961,temperature=0.3
VSKGEELFTGVVPILVELDGDVNGHKFSVKGEGEGDARYGKLTLK...   ← design 1
...
```

Each header encodes:
- `charge`: net charge of the designed sequence
- `score`: ProteinMPNN NLL over the redesigned surface positions
- `global_score`: ProteinMPNN NLL over the full sequence
- `temperature`: sampling temperature used

### Parsed cache

After the first run, surface-residue indices are saved to `parsed/eGFP_seq_indices.pkl`. Subsequent runs reuse this cache, skipping the PyRosetta surface-detection step. Delete the `parsed/` directory to force recomputation.

---

## Optional: Thread sequences onto the backbone

To build relaxed 3D structures from the designed sequences:

```bash
python protein_mpnn_supercharge.py \
    -i examples/eGFP/pdb \
    -o examples/eGFP/eGFP_sc10.fa \
    -c 10 \
    -n 5 \
    --path_to_weights /path/to/ProteinMPNN \
    -mhbond \
    -u \
    --thread \
    --thread_dir examples/eGFP/threaded \
    --thread_workers 4
```

Or, if you already have the FASTA, use `threading_only.py`:

```bash
python threading_only.py \
    -i examples/eGFP/pdb/eGFP.pdb \
    -f examples/eGFP/eGFP_sc10.fa \
    -o examples/eGFP/threaded \
    --num_relax 5 \
    --workers 4 \
    -v
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'protein_mpnn_utils'`**  
The `--path_to_weights` directory does not contain `protein_mpnn_utils.py`, or the path is wrong. Check that the file exists at `<path_to_weights>/protein_mpnn_utils.py` and that it contains the `supercharge_sample` method.

**`ModuleNotFoundError: No module named 'pyrosetta'`**  
PyRosetta is not installed in the active environment. The script will print a clear error and exit. Install it via the wheel from [pyrosetta.org/downloads](https://www.pyrosetta.org/downloads). PyRosetta is only required when no surface-residue cache (`parsed/eGFP_seq_indices.pkl`) exists, or when `--thread` is passed.

**Target charge not reached / sequences capped before target**  
Add `-u` (`--unrestrict`) to allow the temperature to rise automatically. For very large charge changes, higher temperatures (0.5–0.9) may be needed; set `-t 0.5` as the starting temperature.

**`[ WARNING ] missing heavyatom: OXT`**  
This is a harmless PyRosetta warning about a missing C-terminal oxygen. It does not affect surface detection or ProteinMPNN inference.

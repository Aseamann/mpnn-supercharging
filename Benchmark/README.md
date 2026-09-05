# Supercharging benchmark

This directory holds the benchmark that answers the three peer reviews of the
`mpnn-supercharging` manuscript. It measures charge-targeting accuracy,
mutational efficiency, sequence diversity, energetics, hydrogen bonding and
predicted structure for ProteinMPNN-based supercharging against two classical
baselines, a vanilla rejection-sampling ablation and a random-charge control,
across 25 scaffolds spanning three CATH fold classes plus eGFP.

`PLAN.md` at the repository root is the specification. `PROGRESS.md` records what
has actually been run. `logs/ISSUES.md` records every deviation, bug and triage
decision, and should be read before trusting any number in `results/`.
`VERIFICATION_REPORT.md` is the closing check of PLAN.md Section 14 against the
files on disk, and lists what is still open.

## What is here

```
config/          benchmark.yaml (parameters), cluster.yaml (the only file a new user edits)
scripts/         00 to 12, one per phase, plus scripts/lib/ for shared code
slurm/           one sbatch template per phase, and a task file for phases 1 to 5.
                 Phase 6 has none: ESMFold2 shards its FASTA by array index and
                 AF3 reads its own subset CSV
data/            cleaned scaffolds, the manifest, the LayerSelector cache
designs/         one FASTA per (scaffold, method, target)
baselines/       PyRosetta Supercharge outputs
threaded/        relaxed PDBs, per tier
predictions/     ESMFold2 and AlphaFold3 outputs
results/         the CSVs everything downstream reads
notebooks/       analysis.ipynb and analysis_updated.ipynb, plus their modules
figures/         PNG and PDF at 300 dpi; figures/updated/ for the second notebook
tables/          CSV and LaTeX
```

### `figures/` and `tables/` are one run behind `results/`, on purpose

`results/*.csv` describes all nine arms: `designs.csv` is 15,120 rows and
`statistics.csv` is 728 tests. The contents of `figures/` and `tables/` were
produced on 2026-08-26 from the six-arm run, when `designs.csv` was 10,560 rows
and `statistics.csv` 455 tests, and they have deliberately **not** been
regenerated.

Regenerating them would have changed the figures and tables already verified in
PLAN.md Section 14.1 and cited from the manuscript, as a side effect of adding
arms rather than as a decision. So a number read from `tables/T2` will not match
one recomputed from `results/designs.csv`, and the difference is the three new
arms, not an error.

`figures/updated/` is current: it was written from the nine-arm CSVs.

To bring the originals up to date, deliberately:

```bash
python scripts/run_notebook.py                 # rewrites figures/ and tables/
```

### Two notebooks

`analysis.ipynb` is the full set: F1 to F14 plus F2_inset and F8alt, and every
table T1 to T7b. It is unchanged since 2026-08-26 and its outputs are in
`figures/` and `tables/`.

`analysis_updated.ipynb` is a focused second view added 2026-09-05, carrying
ten figures over an expanded arm set and writing to `figures/updated/`. It
exists as a second notebook rather than as edits to the first so the figures
already verified stay reproducible. It also carries draft Methods and Results
paragraphs in its last section. See PLAN.md Section 11.4.

Its figures are numbered **F1 to F10 in plotting order**, a different scheme
from `analysis.ipynb`. The two sets live in separate directories and never
overwrite each other, but a bare "F4" is ambiguous, so the key is:

| `analysis_updated.ipynb` | Content | `analysis.ipynb` |
|---|---|---|
| F1 | Charge ladder against the unguided sampling distribution | F3 |
| F2 | Mutational efficiency | F4 |
| F3 | Sequence diversity | F5 |
| F4 | Energetic cost | F6 |
| F5 | Per-term energy decomposition | F7 |
| F6 | Hydrogen bonds gained and lost | F9 |
| F7 | Independent structure prediction | F10 |
| F8 | Runtime per accepted design | F12 |
| F9 | Where predicted confidence starts to fall | F14 |
| F10 | Charge demand, decoding temperature, predicted confidence | new |


### The nine arms

Six were in the original run. Three were added 2026-09-05:

| Arm | What it is |
|---|---|
| `mpnn_soluble` | primary: supercharging decoder, SolubleMPNN weights |
| `mpnn_vanilla_weights` | same decoder, stock ProteinMPNN weights |
| `mpnn_soluble_hbond_protected` | primary with h-bonded sidechains protected |
| `mpnn_hyper` | **new.** Same decoder, HyperMPNN checkpoint |
| `mpnn_halo` | **new.** Same decoder, HaloMPNN checkpoint |
| `avnapsa` | PyRosetta Supercharge, AvNAPSA mode |
| `rosetta` | PyRosetta Supercharge, score-based mode |
| `rosetta_hbond_off` | **new.** `rosetta` with h-bond protection off, matching the primary MPNN arm |
| `random_control` | random charged substitutions to the same target |

The two biased checkpoints are reached without modifying the repo-root
`protein_mpnn_supercharge.py`, which cannot load an arbitrary `.pt`. Each arm
points `--path_to_weights` at a shim directory under `data/altweights/`
containing symlinks laid out the way that script expects. `logs/ISSUES.md`
records why and what was checked.

The checkpoints themselves live at the repository root in `AlternativeWeights/`:

| File | sha256 |
|---|---|
| `AlternativeWeights/HyperMPNN/v48_020_epoch300_hyper.pt` | `2635afa506cbad850af3fffa7629d80b72ee166d386e2d63a05435b5bd9b459f` |
| `AlternativeWeights/HaloMPNN/epoch_last.pt` | `85c8544b4b8eb7c017c72965bd4c9edbd2919c3c9e2abd47d11054c244f429c8` |

`data/altweights/` is gitignored along with the rest of `data/`, so rebuild the
two shim directories after cloning. Each needs the checkpoint under the name
`v_48_020.pt` inside a `vanilla_model_weights/` subdirectory, and a copy of the
patched `protein_mpnn_utils.py`, because `--path_to_weights` is also the import
root for that module:

```bash
cd Benchmark
PMPNN=$(python -c "import yaml;print(yaml.safe_load(open('config/cluster.yaml'))['paths']['proteinmpnn'])")
ALT=$(python -c "import yaml;print(yaml.safe_load(open('config/cluster.yaml'))['paths']['repo_root'])")/AlternativeWeights
for pair in "hyper:HyperMPNN/v48_020_epoch300_hyper.pt" "halo:HaloMPNN/epoch_last.pt"; do
    name=${pair%%:*}; ckpt=${pair#*:}
    mkdir -p "data/altweights/$name/vanilla_model_weights"
    ln -sfn "$PMPNN/protein_mpnn_utils.py" "data/altweights/$name/protein_mpnn_utils.py"
    ln -sfn "$ALT/$ckpt" "data/altweights/$name/vanilla_model_weights/v_48_020.pt"
done
```

## Reproducing the run

Everything runs on the Rutgers Amarel cluster through Slurm. **Edit
`config/cluster.yaml` first**: it is the only file carrying machine-specific
paths, partitions, walltimes and the bad-node exclusion list.

```bash
source /projects/community/anaconda/2022.10/bd387/etc/profile.d/conda.sh
conda activate /projects/f_sdk94_1/conda/envs/py311
cd /projects/f_sdk94_1/als515/mpnn-supercharging/Benchmark
bash run_all.sh --dry-run     # prints every command without submitting
```

`run_all.sh` is a readable ordered list of the phases, not an unattended driver.
Each phase submits a Slurm array and must drain before the next starts, and
several phases require a timing task to be inspected before the full array is
sized. Run it phase by phase.

Every script takes `--dry-run` and prints exactly what it would read, write or
submit. Failed array cells are rerun by index, and seeds derive from
`(scaffold, method, target)`, so a rerun of one index reproduces its original
run:

```bash
python scripts/02_run_mpnn_supercharge.py --submit --array=17,43,91
```

## Things a reader should know before using the numbers

- **`-mhbond` controls h-bond protection.** Absent, h-bonded side chains are
  protected; passed, they are allowed to mutate. The primary arm passes it,
  because that is how the manuscript's results were produced. The frozen
  `hbond_filter` column is the inverse of the flag and is derived in exactly one
  place, `scripts/lib/io.py`.
- **Input PDBs must be renumbered 1..N.** The repository's `surface_selector`
  returns PDB numbering and then indexes `pose.sequence()` with it, which agree
  only when the input is numbered from 1. Twenty of 25 raw scaffolds crashed and
  two more silently applied Gly/Pro/Cys protection to the wrong residues. eGFP is
  numbered 1..231, so the manuscript's published results are unaffected, but
  anyone applying the method to a fresh PDB download should renumber first.
  Crystal numbering is preserved in `data/scaffolds/<id>/<id>_numbering.csv`.
- **The wild-type energetic reference is relaxed here, not by
  `threading_only.py`**, which does not relax a zero-mutation sequence. All ΔREU
  values are against that relaxed reference. See `logs/ISSUES.md` item 11 for the
  size of the effect and for the one respect in which the reference protocol is
  not identical to the design protocol.
- **Net charge is `(K+R) - (D+E)`**, no histidine and no terminal charges,
  computed in exactly one place, `scripts/lib/charge.py`.

## Reproducing only the analysis

The notebook needs no cluster access and no GPU. Given `results/*.csv` it runs
top to bottom on a fresh kernel and regenerates every figure and table in about
20 seconds.

Two routes, both producing byte-identical tables. As of 2026-08-26 `py311`, the
env every sbatch script uses, has jupyter, nbconvert, nbformat and ipykernel, so
the nbconvert route runs there directly. It takes about 50 s and regenerates
every figure and table:

```bash
conda activate /projects/f_sdk94_1/conda/envs/py311
cd notebooks && jupyter nbconvert --to notebook --execute analysis.ipynb
cd notebooks && jupyter nbconvert --to notebook --execute analysis_updated.ipynb
```

The no-jupyter route takes the notebook as an argument:

```bash
python scripts/run_notebook.py --notebook notebooks/analysis_updated.ipynb
```

`shared_als515` also has a working notebook setup and a wider set of plotting
packages, and is the alternative if `py311` is busy.

`shared_als515` has no torch or PyRosetta and cannot run any other phase, but
the notebook needs neither. From any env with numpy, pandas and matplotlib but
no jupyter, use the bundled runner, which executes the code cells in order in
one namespace and stops at the first failure:

```bash
python scripts/run_notebook.py
```

The two envs are far apart: `py311` is Python 3.11.8 with numpy 1.26.4 and
pandas 2.2.1, `shared_als515` is Python 3.13.5 with numpy 2.3.1 and pandas
2.3.3. All seven table CSVs came out byte-identical across them, so nothing in
the analysis depends on those versions. The figures and tables in the
repository were written from `py311`.

The notebook is stored without cell outputs, so a reader gets the figures by
running it rather than by trusting what is checked in.

It reads only `results/*.csv` and `data/scaffold_manifest.csv`, and writes only
`figures/` and `tables/`.

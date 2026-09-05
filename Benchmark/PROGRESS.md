# Progress

Working notes for picking this up in a new session. `PLAN.md` at the repo root
is the specification. This file records only what has actually been done, what
is blocked, and the exact command to run next.

Last updated: 2026-08-26. All nine phases complete. The AvNAPSA arm was
rerun at `nstruct 10` on 2026-08-26 and phases 5 to 8 were rerun on top of it;
F14 and T7 were added. See `logs/ISSUES.md` `## Instructions taken 2026-08-26`.

## Where things stand

| Phase | State |
|---|---|
| 1. Scaffold curation | **Complete.** 25 scaffolds, meets amended acceptance. |
| 1. LayerSelector cache | **Complete.** 25/25 primary, 7/7 control arm, no failures. |
| 2. Charge ladder | **Complete.** 456/456 cells, 4,560 designs, 100% success. |
| 3. Classical baselines | **Complete.** 400/400 cells, both arms on the full grid. AvNAPSA rerun at `nstruct 10` on 2026-08-26, 200/200 ok, all reproduce `s00`. |
| 4a. Rejection ablation | **Complete.** 25/25 pools, 50,000 samples, 0 failures. Curve written. |
| 4b. Random control | **Complete.** 200/200 cells, 2,000 designs, 0 failures. |
| 5. Threading and energetics | **Complete.** 577/577 records, 1,992 designs, 0 failures. 104 avnapsa cells rethreaded 2026-08-26. |
| 6a. ESMFold2 | **Complete.** 10,585/10,585 predictions, 100%, acceptance is >= 95%. |
| 6b. AlphaFold3 | **Complete.** 36/36, single-sequence mode. See the caveat in ISSUES. |
| 7. Aggregation and statistics | **Complete.** `designs.csv` 10,560 rows, `statistics.csv` 455 tests, 179 significant at Holm-corrected p < 0.05. |
| 8. Notebook, figures, tables | **Complete.** 16 figures, 9 tables, all 28 code cells run clean under `py311` nbconvert. |
| 9. Repo hygiene | **Complete.** Section 14 checklist verified, all 10 items. |

## Decision B is RESOLVED

PyRosetta does expose the supercharge protocol. Verified 2026-08-13 in `py311`:

```python
from pyrosetta.rosetta.protocols.design_opt import Supercharge
```

imports and instantiates, and carries all 22 methods the reference script and
PLAN.md Section 6 use, including `AvNAPSA_positive`, `AvNAPSA_negative`,
`surface_atom_cutoff`, `surface_residue_cutoff` and `get_net_charge`. The
Rosetta baseline does not have to be cut under Section 12 item 4, and AvNAPSA
does not need reimplementing, so it is the genuine protocol and is labelled
`avnapsa`, not `avnapsa_reimpl`. PLAN.md's requirement to validate a
reimplementation against AscG-30 and AscG+36 therefore does not apply.

Both baselines drive the mover through
`Former_Methods/eGFP/run_pyrosetta_supercharge.py`, which is NOT modified.

### Two Phase 3 decisions taken 2026-08-13

**Direction comes from `delta_q`, not the sign of the target charge.**
`run_pyrosetta_supercharge.py:76` and `:91` choose positive or negative
supercharging with `if charge >= 0`, where `charge` is the absolute target. That
equals "raise or lower the charge" only when WT charge and target sit on
opposite sides of zero. On this grid it contradicts the required direction in
**14 of 200 cells per arm**: for example `1a1x_A` has q_wt -7 and target -3, so
the charge must be RAISED by 4, but the sign test picks negative supercharging,
which can only lower it, making the target unreachable by construction. Those
cells would have failed for harness reasons and read as the baseline
underperforming. Direction is taken from `sign(delta_q)`; every other mover
setting is the reference script's, value for value.

**`surface_residue_cutoff(16)` on the Rosetta arm.** PLAN.md Section 6 names
`-surface_atom_cutoff 120`, but in the Supercharge mover that parameter governs
the AvNAPSA sequence-based surface definition while `surface_residue_cutoff`
(a neighbour count) governs score-based Rosetta mode. PLAN.md conflated them.
The reference script uses the mode-correct one; both are Rosetta defaults.
Both decisions are now recorded in PLAN.md Section 0.1 as items 14 and 15, and
Section 6 is amended in place.

## Phase 1 results

25 scaffolds: 8 `mainly_alpha`, 8 `mainly_beta`, 8 `alpha_beta`, plus eGFP as
the focus scaffold. Lengths 80 to 231, WT net charge -9 to +14 with 7 positive.
`n_designable` 47 to 96, median 63.

Curation funnel: 1120 CATH 4.2 test chains -> 1054 after `excluded_PDBs.csv`
-> 1054 fetched with 0 failures -> 479 passed the Section 4 filters -> 407
cluster representatives at 30% identity -> 377 classified -> 25 selected.

`loop_rich` was dropped (Decision 11) and the chain filter was relaxed to the
designed chain (Decision 12). Both are recorded in PLAN.md Section 0.1 and
Section 4 was amended to match.

## Phase 2 results

| Arm | Cells | Designs | Weights | `-mhbond` | Working dir |
|---|---|---|---|---|---|
| mpnn_soluble | 200 | 2000 | soluble | passed | `data/` |
| mpnn_vanilla_weights | 200 | 2000 | original | passed | `data/` |
| mpnn_soluble_hbond_protected | 56 | 560 | soluble | absent | `data/hbond_protected/` |

456/456 cells succeeded against a >= 95% acceptance bar. Runtime 2.9 to 76.5 s
per cell, median 9.2 s, on `main-redhat` with 4 CPUs.

Temperature under `-u`: 102 of 456 cells escalated above 0.3, up to 19
escalations, and none exceeded the 0.9 ceiling. Final temperature is 0.3 for
354 cells with the remainder spread across 0.4 to 0.9. Decision E defines F11's
ceiling as the `final_temperature == 0.3` cells, so that column carries real
signal rather than being degenerate.

Verified rather than assumed: every FASTA parses under `lib/io.py`, every
design's header charge agrees with an independent recomputation through
`lib/charge.py`, and every native charge agrees with the manifest `q_wt`.

Outputs are `designs/*.fa` and per-cell sidecars in `results/cells/*.json`,
aggregated into `results/designs.csv` by Phase 7.

## Phase 4 results

Built as `scripts/lib/phase4.py` plus the two thin drivers
`05_run_vanilla_rejection.py` and `06_run_random_control.py`,
`slurm/array_phase4.sbatch`, and `slurm/tasks_phase4.tsv` (50 tasks: rejection
at indices 0 to 24, random control at 25 to 49). Tasks are per scaffold, not per
cell: one rejection pool serves all 8 of that scaffold's targets, and a control
task walks its 8 targets in about 4 s.

### 4b random control: complete

200/200 cells, 2,000 designs, 0 failures, 0.1 s of compute in total. Every
replicate reaches its target exactly and no replicate ran out of designable
positions, on any scaffold, at any rung including +/-24 density.

Mutation counts are **identical to `mpnn_soluble`**: median 11.0, mean 13.25 for
both arms. That is the expected result, not a bug. The number of mutations is
fixed by the charge arithmetic and the designable set, both of which are held
equal, so the control cannot be separated from the real method on hit rate,
mutation count, or charge. The separation has to come from Phase 5 energetics and
Phase 6 structure prediction, which is what the control exists to test. All 10
replicates of every cell are distinct mutation sets.

### 4a rejection: complete

25/25 pools, 50,000 samples, 0 failures. `results/rejection_curve.csv` has 200
rows, 25 scaffolds x 8 targets, and `results/rejection_distribution.csv` has the
per-scaffold mean and sd Section 7 asks for and the curve schema has no room for.
Section 7 acceptance is met. Zero-hit cells are written censored as `>2000`,
never extrapolated.

Timing: eGFP, the worst case at 231 residues, took 627.24 s for its 2,000
samples on `main-redhat` with 4 CPUs, 0.314 s per sample; the shortest scaffolds
run at 0.135 s per sample. `phase4_rejection.time` is set from that at one hour.

**Coverage collapse, by ladder rung.** Each cell is 2,000 vanilla samples.

| ΔQ density | cells with 0 hits | total on-target draws | best cell |
|---|---|---|---|
| -24 | 22/25 (88%) | 236 | 233 |
| -16 | 19/25 (76%) | 314 | 222 |
| -8 | 7/25 (28%) | 1,138 | 257 |
| -4 | 1/25 (4%) | 3,190 | 276 |
| +4 | 8/25 (32%) | 1,362 | 245 |
| +8 | 12/25 (48%) | 188 | 56 |
| +16 | **25/25 (100%)** | **0** | 0 |
| +24 | **25/25 (100%)** | **0** | 0 |

**119 of 200 cells get zero on-target samples out of 2,000.** At +16 and +24
density it is 50 of 50: across 50,000 vanilla samples spanning every scaffold,
not one landed on a positive supercharging target.

**The reason is a systematic negative bias in the base model, not scaffold
noise.** Vanilla soluble ProteinMPNN pulls every scaffold toward roughly -5
regardless of where it started. `3sjq_C` has `q_wt` +14 and samples at -4.78;
`4o32_C` has `q_wt` +10 and samples at -3.02. Only 2 of 25 scaffolds sample more
positive than their wild type. Sampling sd is 2.0 to 5.0 across scaffolds, so the
positive rungs sit 6 to 22 sd out. That is why rejection sampling cannot reach
them at any sample budget worth spending, and it is the quantitative form of the
Section 7 argument: the method is an importance sampler for a region the base
model essentially never visits.

**Cost per on-target design, CPU seconds** (Decision D: the `gpu_seconds_*`
columns hold CPU seconds under legacy names). Supercharging cost is the Phase 2
`mpnn_soluble` cell's wall time divided by its on-target designs, recounted from
the FASTA through `lib/charge.py`.

| | rejection | supercharging | ratio |
|---|---|---|---|
| 81 cells where rejection found the target | median 8.6 s | median 0.62 s | median 15x, max 834x |
| 119 cells where it found nothing | median >411 s | median 1.33 s | unbounded |

Note the ratio is not the headline. In 119 of 200 cells rejection sampling has
no measurable cost per hit because it never produced a hit, so those rows are
censored lower bounds and the comparison there is qualitative, not a speedup.

## Phases 5 to 9

### Phase 5: threading and energetics

`scripts/lib/rosetta_metrics.py`, `scripts/07_thread_and_score.py`,
`slurm/array_thread.sbatch`, `slurm/tasks_phase5.tsv` (577 tasks: 25 wild-type
references, 552 design cells, 1,992 designs since the 2026-08-26 AvNAPSA rerun,
1,728 before it).

Tier A is eGFP at every method, target and sample, `num_relax 5`. Tier B is the
other 24 scaffolds at densities +/-8 and +/-16, first 3 samples, `num_relax 2`.
eGFP is deliberately NOT re-threaded in tier B: it is already covered at the
deeper tier, and `designs.csv` holds one `thread_tier` per design.

Timing on `main-redhat` with 4 CPUs: a tier A cell (eGFP, 10 designs,
`num_relax 5`) took 5,641 s; a tier B wild-type reference took a median 115 s.

**The wild-type reference had to be built here.** `threading_only.py:234-239`
does not relax a zero-mutation sequence, it dumps and scores the crystal pose.
Measured relaxation gain across the panel: **median 305.5 REU, range 155 to
671**. Referencing designs against the unrelaxed crystal would have made every
design look about 1 to 3 REU per residue better than wild type from relaxation
alone. See `logs/ISSUES.md` item 11, including the caveat that the reference
relaxes unrestricted while designs relax within a shell.

### Phase 6: structure prediction

ESMFold2 is **complete: 10,585/10,585 records, 100% ok** against a >= 95% bar
(8,785 before the 2026-08-26 AvNAPSA rerun; the 1,800 new sequences were folded
by resubmitting the same 40 shards, which skip records whose `.cif` exists).
Median TM-score to the wild-type crystal 0.886, median CA RMSD 1.57 A. Model is
`biohub/ESMFold2-Fast` at the tool's defaults, recorded in every row; the
accuracy check is the AF3 subset, per the tool README's own advice.

AF3 runs 36 jobs on `p_sdk94_1` in single-sequence mode (empty MSAs, no
templates, `--norun_data_pipeline`). Subset is eGFP plus the median-WT-pLDDT
scaffold in each fold class: `4o32_C`, `3iu6_A`, `1at0_A`. That is 4 scaffolds,
not Section 9's 5, because Decision 11 left 3 fold classes rather than 4.

### Phase 7: aggregation and statistics

`results/designs.csv` has 10,560 rows on the frozen 47-column schema, every row
validated, `status="ok"` throughout and no duplicate `design_id`.
`statistics.csv` 455 tests of which 179 are significant at Holm-corrected
p < 0.05. 144 of the 455 carry an empty CI: BCa needs variation to jackknife and
refuses a degenerate sample rather than fabricating an interval
(`11_statistics.py:118`). Those rows still carry `median_diff` and `p_holm`.
Before the 2026-08-26 AvNAPSA rerun this was 8,760 rows and 437 tests.

Headline paired results against `mpnn_soluble`, pooled over all strata:

Recomputed 2026-08-26 from the 455-test `statistics.csv`, after the AvNAPSA
rerun. `n_pairs` is 200 for each of these.

| metric | vs avnapsa | vs rosetta | vs random_control |
|---|---|---|---|
| hit rate, Cliff's delta | 0.49 | 0.99 | 0.00 |
| mutations per charge, median diff | +0.234 (0.219 to 0.250), delta 0.94 | +0.063 (0.048 to 0.083), delta 0.39 | 0.00 |
| unique mutation sets, median diff | +9, delta 1.00 | +3, delta 0.86 | 0 |
| Δ pLDDT vs WT prediction | -0.44 (-0.74 to -0.02) | -1.21 (-1.70 to -0.79) | **+1.32 (0.95 to 1.56)** |

The unique-mutation-sets row is now a measurement on both sides. At `nstruct 1`
it was an artefact of the design matrix; at `nstruct 10` AvNAPSA returns a
median of 1 distinct set out of 10 against `mpnn_soluble`'s 10, Cliff's delta
1.00, so the separation is complete rather than constructed.

Two of these matter for the response letter and point in opposite directions.
**The method costs more mutations per unit charge than AvNAPSA**, 0.234 more,
with a tight CI and Cliff's delta 0.94; that is a real weakness and Section 10.3
requires reporting it. **The method beats the random-charge control on predicted
structure by 1.32 pLDDT** (p = 1.1e-14), which is the first metric on which the
control separates from the method at all and is what the control was built to
test: charge arithmetic alone does not reproduce it.

### Phases 8 and 9

`notebooks/analysis.ipynb` (38 cells) with `analysis_lib.py` and `tables_lib.py`.
All 13 figure functions render. `README.md`, `.gitignore`,
`MANUSCRIPT_CHECKLIST.md` and a refreshed `results/environment.json` are written.

The figure palette is measured, not eyeballed: `scripts/validate_palette.py` is a
Python port of the `dataviz` skill's Node validator, which Amarel cannot run. It
reproduces that validator's documented numbers exactly. The six method colours
pass for lines, bars and boxplots (worst protan/deutan dE 9.1) and **fail** for
scatter overlays (3.2), so F10 facets by method instead.

## Environment

Activate before anything:

```bash
source /projects/community/anaconda/2022.10/bd387/etc/profile.d/conda.sh
conda activate /projects/f_sdk94_1/conda/envs/py311
cd /projects/f_sdk94_1/als515/mpnn-supercharging/Benchmark
```

- `tmtools` 0.3.0 pip-installed into `py311` for Phase 6 TM-align.
- **DSSP lives in its own env**, `/projects/f_sdk94_1/conda/envs/dssp`, pointed
  at by `paths.dssp_bin`. It cannot live in `py311`: the salilab `dssp 3.0.0`
  package does not declare its boost dependency, so `boost-cpp=1.73` must be
  installed explicitly alongside it, and that is unsatisfiable against py311's
  Python 3.11 pin. Verified emitting genuine 8-state codes including G.
- The `py311` footprint changed slightly and not entirely reversibly: an earlier
  DSSP attempt there pulled `libgcc-ng` 13.2.0 to 16.1.0 and updated `openssl`,
  `certifi` and `ca-certificates`. The `dssp` and `libboost` packages were
  removed again. numpy stayed 1.26.4 and PyRosetta stayed 2024.19 throughout,
  and all imports plus `pyrosetta.init()` were re-verified afterwards.
- `results/environment.json` is written by `scripts/capture_environment.py`.
  Rerun it after any install so the record stays true.

## Phase 3 results, all 400 cells

400/400 succeeded, no reruns needed. Both arms cover the same 200 scaffold x
target grid as `mpnn_soluble`, exactly, with every FASTA, PDB and sidecar on
disk. Section 6 acceptance met. `mpnn_soluble` is shown for scale only; the
comparison proper is Phase 7.

| | avnapsa | rosetta | mpnn_soluble |
|---|---|---|---|
| cells | 200 | 200 | 200 |
| replicates per cell | 1 | 10 | 10 |
| hit-exact, replicate level | 118/200 = 59.0% | 464/2000 = 23.2% | 2000/2000 = 100% |
| cells with every replicate on target | 59.0% | 1.0% | 96.0% |
| cells with >= 1 replicate on target | 59.0% | 70.5% | 100% |
| n_mutations, median / mean | 8.5 / 9.96 | 9.0 / 10.45 | 11.0 / 13.25 |
| n_mutations, min to max | 2 to 35 | 1 to 46 | 2 to 50 |
| median n_unique_mutation_sets | 1 (forced) | 7 | 10 |
| median cell wall time | 16.5 s | 237.5 s | 9.4 s |

Hit rate by |ΔQ| bucket, replicate level:

| bucket | avnapsa | rosetta |
|---|---|---|
| 1 to 5 | 61% | 30% |
| 6 to 10 | 73% | 36% |
| 11 to 20 | 59% | 23% |
| 21+ | 48% | 11% |

Misses run in opposite directions. Of 82 avnapsa misses, 73 overshoot and 9
undershoot. Of 1,536 rosetta misses, 976 undershoot and 560 overshoot.

**Both of PLAN.md Section 5's determinism assumptions are contradicted, and
neither has been acted on.** Rosetta is not near-deterministic: the median cell
gives 7 distinct mutation sets out of 10, mean 6.6, and only 3 of 200 cells give
one. AvNAPSA's determinism remains **unmeasured**, and this run cannot measure
it: `nstruct` is 1, so `n_unique_mutation_sets` is 1 in all 200 cells by
construction rather than by observation, and must not be read as confirming the
assumption. The reference runs in `Former_Methods/eGFP/TargetPos2_Avn/` gave 65,
65 and 61 mutations across three replicates, pointing the other way. At a median
16.5 s per cell, each extra replicate across the arm costs about 12 CPU minutes.
Raising `nstruct` is a schema change and needs PLAN.md Section 5 amended first.

## Phase 3 early signals, from the two timing cells

Both on `1a1x_A` at target -32 (q_wt -7, the most extreme rung), so treat as
one hard cell rather than a general result until all 400 land.

- avnapsa: 17.1 s, hit -32 exactly, 17 mutations.
- rosetta: 116.3 s for 10 replicates, **0/10 hit target** (reached -18 to -20,
  short by 12 to 14) with only 10 to 12 mutations, and **6 distinct mutation
  sets out of 10**.
- For comparison the mpnn_soluble cell hit -32 exactly with no temperature
  escalation.

Two PLAN.md assumptions look wrong and are now being measured rather than
assumed: Section 5 calls AvNAPSA "deterministic by construction" (the reference
runs in `Former_Methods/eGFP/TargetPos2_Avn/` gave 65, 65 and 61 mutations), and
frames `nstruct 10` as testing whether Rosetta is near-deterministic (6/10
distinct here).

## Bugs found and fixed

Full detail in `logs/ISSUES.md`. Ordered by how much damage they would have done.

1. **`surface_selector` mixes PDB and pose numbering** (in the repo script,
   not the benchmark). Line 84 returns PDB numbering, line 412 indexes
   `pose.sequence()` with it. They agree only when the input is numbered 1..N.
   20 of 25 scaffolds crashed; 2 more were shifted but in range and silently
   applied the Gly/Pro/Cys protection to the wrong residues. eGFP is numbered
   1..231 so the manuscript's published results are unaffected. Fixed on the
   benchmark side by renumbering cleaned scaffolds to 1..N; crystal numbering is
   preserved in `data/scaffolds/<id>/<id>_numbering.csv`. The repo script is not
   modified.
2. **The exclusion list matched nothing.** `excluded_PDBs.csv` is a pandas dump
   whose first column is a row index; reading `row[0]` built a set of integers,
   so 0 of 1120 chains were excluded. Now read by the `PDB_IDS` column.
3. **Every resolution parsed as 2.0 A**, because the first float on the
   `REMARK   2 RESOLUTION.` line is the remark number. Every structure would
   have passed the 2.5 A cutoff on a fabricated value.
4. **`-run:jran` overflowed int32.** `lib/io.cell_seed` returns up to 2**32-2
   but PyRosetta parses `-run:jran` as signed 32-bit, so **205 of 400** Phase 3
   cells would have died with "Illegal value for integer option". Only caught
   because the two timing cells drew seeds on opposite sides of the boundary.
   Fixed by folding the seed for PyRosetta rather than narrowing `cell_seed`,
   which would have silently changed the seeds already recorded in the finished
   Phase 2 runs. The sidecar records both `seed` and `pyrosetta_jran`.
5. **`is_native` misread baseline FASTAs.** It keyed off `temperature is None`,
   but the Supercharge mover emits no temperature on any record, so every
   baseline record looked native and `split_native` rejected the file. Now keyed
   off a trailing `_<digits>` in the name. All 456 Phase 2 FASTAs reverified.
6. **`--chain_id` was never passed** in Phase 2, defaulting to `'A'`, so the six
   non-A-chain scaffolds admitted by Decision 12 all failed with
   `KeyError: 'seq_chain_A'`. 96 of 456 cells. Latent until the chain filter was
   relaxed.
7. **Multi-model entries counted once per model** (NMR 1a90 chain A read as
   3348 residues). Parsing stops at `ENDMDL`.
8. **`lib/io.py` shadowed the stdlib `io`**; `lib/` is now a package.
9. **`sbatch` could not find the repo** because it spools the script;
   drivers now pass `--export=BM_ROOT=`.
10. **`conda.sh` path was wrong**; the base is the community anaconda module.

## Discrepancies carried into Phase 9

- `-mhbond` is passed in the primary arm, which is how the manuscript's results
  were produced. Absent, h-bonded sidechains are protected; passed, they may
  mutate. `hbond_filter` is the inverse of the flag. Flag and documentation
  stand as they are (author's decision, 2026-08-26).
- The repo script's LayerSelector cache key does not encode
  `mutate_hbonded_sidechains` even though the cached result depends on it. Each
  arm therefore gets its own working directory. Measured effect: eGFP has 96
  designable positions with `-mhbond` and 62 with protection on, so a shared
  cache would have made the control arm measure nothing.
- The numbering bug in item 1 above is worth a line in the README, since it
  affects anyone applying the method to a PDB download.

## Restarting

Everything through Phase 2 is complete and on disk. Do not rerun it.

Always start with:

```bash
source /projects/community/anaconda/2022.10/bd387/etc/profile.d/conda.sh
conda activate /projects/f_sdk94_1/conda/envs/py311
cd /projects/f_sdk94_1/als515/mpnn-supercharging/Benchmark
```

### Phase 3: check the arrays, then verify

Phase 3 is fully built and both arrays were submitted. Nothing needs writing.
Files: `scripts/lib/baseline.py`, `scripts/03_run_avnapsa.py`,
`scripts/04_run_rosetta_supercharge.py`, `slurm/array_baseline.sbatch`,
`slurm/tasks_phase3.tsv` (400 cells), plus `baseline_arms` in `benchmark.yaml`
and `pyrosetta_supercharge_script` / `phase3_baseline` in `cluster.yaml`.

First check whether they finished:

```bash
squeue -u als515 -h -t pending,running -r | wc -l
ls results/cells/*avnapsa*.json | wc -l    # expect 200
ls results/cells/*rosetta*.json  | wc -l    # expect 200
```

Then summarise status, hit rate, mutation counts and
`n_unique_mutation_sets`, and rerun any failed indices:

```bash
python scripts/03_run_avnapsa.py --submit --array=<failed ids>
python scripts/04_run_rosetta_supercharge.py --submit --array=<failed ids>
```

Acceptance (PLAN.md Section 6): baseline outputs exist for the same
scaffold x target grid as the MPNN arms, and mutation lists parse into the same
schema. Outputs are `baselines/<method>/<cell_id>/`, `designs/<cell_id>.fa`
and `results/cells/<cell_id>.json`.

**PLAN.md is now amended** (2026-08-13, approved). Section 0.1 carries the two
Phase 3 decisions as items 14 and 15 and the Decision B resolution as item 13;
item B is struck from Section 0.2 and D is marked resolved with the measured CPU
timing. Section 6 is amended in place, with the superseded reimplementation
requirement kept and marked so the reasoning stays auditable, and with the
`nstruct 10` diversity prediction flagged as contradicted by the measurement.
Standing permission: PLAN.md may be edited without asking.

### Rerunning failed cells in any phase

Task files are stable and seeds derive from (scaffold, method, target), so an
index can be rerun and will reproduce its original run:

```bash
python scripts/02_run_mpnn_supercharge.py --submit --array=17,43,91
python scripts/04_run_rosetta_supercharge.py --submit --array=17,43,91
```

### Phase 4 is complete

Nothing to run. If a pool ever needs regenerating, `--curve` is idempotent and
rebuilds both CSVs from whatever is on disk:

```bash
python scripts/05_run_vanilla_rejection.py --submit --array=<failed ids>
python scripts/05_run_vanilla_rejection.py --curve
```

### Phase 5 is complete

Stale heading corrected 2026-08-14: this said "Not started" long after the phase
finished. `scripts/07_thread_and_score.py` and `scripts/lib/rosetta_metrics.py`
both exist and 577/577 records landed. See the Phase 5 section above for the
tier definitions, the timings and the wild-type reference problem.

The scoping note it carried was acted on: the random control is threaded on the
same subset as `mpnn_soluble`, cell for cell, so Section 7's "same subset used
for energetic scoring" holds and the control stays a control. `d_reu_per_res` is
populated on 1,992 designs, and the paired `mpnn_soluble` against
`random_control` test on `median_d_reu_per_res` runs on 104 pairs.

### Checking what is running

```bash
squeue -u als515 -o "%.10i %.14j %.9P %.8T %.10M %R"
squeue -u als515 -h -t pending,running -r | wc -l   # against the 500 CPU cap
```

## Job history

| Job | What | Outcome |
|---|---|---|
| 60552137 | Phase 1 curate | Failed instantly, `BASH_SOURCE` spool bug. Fixed. |
| 60552173 | Phase 1 curate | 1054 fetched, 188 filtered, 179 representatives. |
| 60552281 | Filter/cluster rerun after ENDMDL fix | Identical results. |
| 60552855 | dssp, select, targets | Ran, but produced a 23-row manifest failing acceptance. |
| 60553349 | Recuration after Decisions 11 and 12 | 479 filtered, 407 representatives, 25-row manifest. |
| 60553411 | LayerSelector timing | Failed on the numbering bug. |
| 60553425 | LayerSelector timing retry | 9.87 s, n_designable 62. |
| 60553431 | LayerSelector array, 31 tasks | All clean. |
| 60553831 | Phase 2 timing cell | 14.98 s, target -32 hit exactly, 0 retries. |
| 60553903 | Phase 2 array, 455 cells | 360 ok, 96 failed on `--chain_id`. |
| 60554396 | Phase 2 rerun of the 96 | All ok. 456/456 complete. |
| 60555863 | Phase 3 avnapsa timing cell | 17.1 s, hit -32 exactly, 17 mutations. |
| 60555864 | Phase 3 rosetta timing cell | Failed: `-run:jran` int32 overflow. |
| 60555878 | Phase 3 rosetta timing retry | 116.3 s, 0/10 hit, 6 unique sets. |
| 60555884 | Phase 3 avnapsa array, 200 cells | All 200 ok. |
| 60555885 | Phase 3 rosetta array, 200 cells | All 200 ok. |
| 60560442 | Phase 4a rejection timing, eGFP | 627.24 s, 2,000 samples, q_mean -15.64. |
| 60560443 | Phase 4b random control, 1 scaffold | 4 s, 8 cells, all ok. |
| 60560467 | Phase 4b random control, 24 scaffolds | All ok. 200/200 cells complete. |
| 60560704 | Phase 4a rejection array, 24 scaffolds | All 24 ok. 25/25 pools complete. |

## Not done

This section was stale as of 2026-08-14 and is rewritten. It previously said the
result CSVs, notebook, figures and tables did not exist, which contradicted the
status table at the top of this file. They all exist and are verified; see
PLAN.md Section 14.1 for the evidence behind each one.

- Nothing in PLAN.md remains unbuilt. The items below are decisions and
  housekeeping, not work.
- **Nothing is committed to git.** `Benchmark/` is still one untracked
  directory. About 2,754 files and 36 MB would be staged, of which 21 MB is
  `logs/mpnn/`. See PLAN.md Section 14.2 item 2 before committing.
- **AvNAPSA `nstruct` is still open.** See `logs/ISSUES.md` `## Open` and
  PLAN.md Section 0.2. It changes the frozen design matrix, so it waits for an
  instruction.
- **`fold_class` is null for the 408 eGFP rows.** By construction. PLAN.md
  Section 14.2 item 1.
- The manuscript-side fixes in `MANUSCRIPT_CHECKLIST.md` are unapplied by
  design: the manuscript is outside `Benchmark/` and out of scope.
- `data/raw/` is 894 MB and gitignored.

## Phase 8 verification, 2026-08-14

The claim "all 26 code cells run clean" that this file previously carried had
not been demonstrated by running the notebook. The notebook was stored with
`execution_count: null` and zero outputs on all 26 code cells, and `py311` has
no jupyter, nbconvert, nbformat or ipykernel, only IPython. Figures and tables
had been produced by calling `analysis_lib` directly.

It has now been run both ways, and the claim holds:

- `jupyter nbconvert --to notebook --execute` under a real ipykernel in
  `/projects/f_sdk94_1/conda/envs/shared_als515`, 35 s, exit 0, no cell errors.
  That env has jupyter, numpy, pandas and matplotlib but no torch or PyRosetta,
  which is enough because the notebook only reads CSVs.
- `python scripts/run_notebook.py` in `py311`, 26/26 cells ok in 11 to 21 s.
  New script, added 2026-08-14, for the no-jupyter route. Takes `--dry-run`.

**Cross-environment check: all seven table CSVs are byte-identical between the
two.** `py311` is Python 3.11.8, numpy 1.26.4, pandas 2.2.1; `shared_als515` is
Python 3.13.5, numpy 2.3.1, pandas 2.3.3. Nothing in the analysis depends on
those versions. The figures and tables in the repository were written from
`py311`.

Also fixed: `README.md` and `run_all.sh` both told the reader to run
`jupyter nbconvert` from `py311`, where it fails with "no such file"; the
notebook's cells had no nbformat `id` fields, which newer nbformat will treat
as a hard error, and are now normalized. The notebook is still stored without
outputs, so a reader gets the figures by running it.

## Three arms added and a second notebook, 2026-09-05

On the author's instruction. PLAN.md Section 0.1 items 17 to 21 record the
decisions, `logs/ISSUES.md` records what had to be worked around.

**Arms.** `mpnn_hyper` and `mpnn_halo` run the supercharging decoder over the
HyperMPNN and HaloMPNN checkpoints on the full 25 x 8 x 10 grid, 2,000 designs
each. `rosetta_hbond_off` is the Rosetta baseline with
`dont_mutate_hbonded_sidechains` set False, on the 7-scaffold control subset,
560 designs. `designs.csv` goes from 10,560 rows to 15,120. No schema change:
new arms are new rows, and PLAN.md Section 10.1 stays frozen.

**The biased checkpoints without patching the repo script.**
`protein_mpnn_supercharge.py` builds its checkpoint path from a fixed directory
layout and has no `--checkpoint` flag, so each biased arm points
`--path_to_weights` at a shim directory under `data/altweights/` holding
symlinks. Checked rather than assumed: both checkpoints load, are
architecturally identical to `v_48_020`, resolve to different files, and produce
sequences that differ from `mpnn_soluble` on the same cell. Identical sequences
would have meant a silent fallback to the stock weights.

**The h-bond asymmetry the hydrogen-bond figure was carrying.** `lib/baseline.py` hardcoded
h-bond protection ON for the Rosetta arm while the primary MPNN arm passes
`-mhbond`, which turns it OFF. The two were being compared anyway. The setting
is now per-arm, defaulting to the old value so `rosetta` is unchanged, and the
new arm closes the comparison. The control-subset selection rule moved into
`lib/io.control_subset` so the Phase 2 and Phase 3 control arms cannot resolve
to different scaffold sets.

**The unguided-MPNN reference was measured with no new sampling.**
`scripts/12_rejection_hit_metrics.py` scores the 6,428 samples already sitting
in the Phase 4a pools that land exactly on a ladder rung, into
`results/rejection_hits.csv`. Both of its limits are recorded: zero on-target
samples exist at +16 and +24 across all 25 scaffolds, while every negative rung
has some, and a cell contributes only where the pool hit at least once, 81 of
200, which makes the unguided cost a lower bound.

**Task-id stability.** New arms were appended, never inserted, because
`--emit-tasks` numbers tasks in arm order. Verified: the first 456 rows of
`tasks_phase2.tsv` and the first 400 of `tasks_phase3.tsv` are unchanged on
every shared column, so `--array=` reruns of the completed arms still address
the same cells. Phase 5's task file does renumber, since it sorts by method, so
the new threading tasks were selected by which cells had no output record rather
than by index arithmetic.

**Second notebook.** `notebooks/analysis_updated.ipynb` and
`analysis_updated_lib.py` carry ten figures over the nine arms and write to
`figures/updated/`. `analysis.ipynb`, `analysis_lib.py`, `tables_lib.py` and the
existing `figures/` and `tables/` are untouched, so the set verified in PLAN.md
Section 14.1 stays reproducible. Explanatory text moved out of figure titles
into per-figure Notes cells.

**Palette.** Nine arms need a bigger palette than the certified six.
Arms are grouped into hue families with lightness separating members, and
`scripts/validate_palette.py` measures it: adjacent pairs 15.9 normal-vision and
14.7 protan/deutan, both PASS, which is more colourblind margin than the
six-slot palette had (9.1). All pairs still fails, as it did before, so no
figure puts arbitrary series side by side.

**`figures/` and `tables/` were not regenerated.** They date from 2026-08-26 and
describe the six-arm run: 10,560 designs, 455 statistical tests. `results/*.csv`
now describes nine arms, 15,120 designs and 728 tests. Regenerating the old
outputs would have silently changed figures already verified and cited, as a
side effect of adding arms rather than as a decision, so they were left alone
and the gap is documented in README.md instead. `figures/updated/` is current.
`scripts/run_notebook.py` with no `--notebook` argument rewrites the originals
when that is wanted.

**Figures in the updated notebook were renumbered F1 to F10 in plotting order,
2026-09-05.** The old numbering was inherited from `analysis.ipynb` and had
gaps, since the updated notebook drops F1, F2, F8, F11 and F13. `figures/` and
`figures/updated/` are separate directories so nothing was overwritten, and
README.md carries the mapping between the two schemes. References in this file
and in `logs/ISSUES.md` dated before 2026-09-05 use the original scheme and
describe `analysis.ipynb`; they were left as written.

**The unguided series was dropped from the mutational-efficiency figure but not
from the results.** It spends 8 to 12 mutations per unit charge against roughly
0.8 for every guided arm, so on a shared axis it flattened the comparison the
figure exists to make. `results/rejection_hits.csv` still holds all 6,428
on-target samples and the notebook reports them per fold class as a table.

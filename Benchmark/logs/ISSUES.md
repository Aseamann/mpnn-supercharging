# Issues log

Problems found, decisions taken, and discrepancies between the code and the
manuscript. Newest entries at the bottom of each section.

## Environment verification

**ProteinMPNN clone, checked 2026-08-13.**

- `grep -n "def supercharge_sample" /home/als515/GitHub_Repos/ProteinMPNN/protein_mpnn_utils.py`
  returns line 1228. The method is present. `Modified_ProteinMPNN/protein_mpnn_utils.py`
  was not copied over it and nothing under `/home/als515/` was modified.
- ProteinMPNN commit: `8907e66`.
- `soluble_model_weights/v_48_020.pt` exists, so the primary arm's weights are available.
- `soluble_model_weights/excluded_PDBs.csv` exists and is used by Phase 1 (see Decision C below).

**Pending installs.** DSSP (`mkdssp`) and TM-align are both absent from
`/projects/f_sdk94_1/conda/envs/py311`. Neither has been installed yet. Phase 1's
`dssp` stage stops with a clear message rather than proceeding without it.
`py311` is shared with other lab members, so only these two will be installed,
and the exact commands and resolved versions go into `results/environment.json`
once run.

## Discrepancies to carry into the Phase 9 README and Methods checklist

**`-mhbond` semantics, recorded so the `hbond_filter` column can be read.**
`protein_mpnn_supercharge.py:585` defines `-mhbond` / `--mutate_hbonded_sidechains`
as `action='store_true', default=False`. With the flag absent the script
protects h-bonded sidechains from mutation; passing `-mhbond` removes that
protection. The script is not being changed. This benchmark passes `-mhbond` in
the primary arm, which is the configuration the manuscript's results were
produced under. 2026-08-26: the author's decision is that the flag and its
documentation stand as they are, so this is a note on semantics and not a
correction to be carried into the manuscript.

The `hbond_filter` column reads backwards from the flag name as a result:
`hbond_filter=False` means `-mhbond` was passed. `lib/io.py:hbond_filter_from_flag`
is the only place that inversion happens.

**The LayerSelector cache key does not encode the h-bond flag.**
`protein_mpnn_supercharge.py:685` reads and writes `parsed/<pdb>_seq_indices.pkl`,
and the comment above it calls this a base cache that "never depends on axis
params". But line 693 calls `parse_for_supercharge(...)` with
`args.mutate_glyprocys`, `args.mutate_hbonded_sidechains`, `args.no_fastrelax`
and `args.add_histidine`, so the cached designable-position set *does* depend on
the h-bond flag while the filename does not.

Left alone, the h-bond-protected control arm described in PLAN.md Section 5
would load the primary arm's cache and silently inherit its designable set,
erasing exactly the difference that arm exists to measure. Handled by giving
each arm its own working directory (`arms[].workdir` in `config/benchmark.yaml`),
so the caches land in `data/parsed/` and `data/hbond_protected/parsed/`
respectively. The repo script is not modified and nothing is symlinked.

**PyRosetta is initialised without `-run:constant_seed` by the repo script.**
`_init_pyrosetta` at line 28 calls `init(silent=True)` with no options, which
does not satisfy PLAN.md ground rule 3. `01_prepare_structures.py` initialises
PyRosetta itself with `-run:constant_seed -constant_seed -jran 20260813 -mute all`
and then sets the repo module's `_pyrosetta_initialised = True` so its own
`init()` is skipped. No change to the repo script.

## Decisions taken

Resolved 2026-08-13, recorded in PLAN.md Section 0.1 and implemented in
`config/benchmark.yaml`.

**A. MPNN seeds (Phase 2).** `protein_mpnn_supercharge.py:642` draws
`seed = np.random.randint(0, high=999)` from an unseeded RNG, never prints it,
and exposes no `--seed`. `scripts/02_run_mpnn_supercharge.py` seeds numpy
immediately before calling `main()`, which makes that draw deterministic. One
seed per cell, derived from `base_seed` and a blake2b digest of
(scaffold, method, target charge), so rerunning a single failed array index
reproduces its original run. The recorded `seed` is the 64-bit value passed to
`np.random.seed`, not the script's internal 0..999 draw, so the collision
concern across 2,640 designs does not apply.

**C. CATH 4.2 test split (Phase 1).** The split is the candidate pool and
`soluble_model_weights/excluded_PDBs.csv` is applied on top of it, because
`mpnn_soluble` is the primary arm. The split is not in the ProteinMPNN clone and
is fetched to `data/splits/`. `00_curate_scaffolds.py` verifies the downloaded
JSON exposes a `test` key and stops if it does not; it never substitutes another
list. The URL in `config/benchmark.yaml` has not yet been exercised, so it is
unconfirmed until the `pool` stage runs for real.

**E. F11 and the `-u` flag (Phase 2).** Phase 2 is a single `-u` run of 2,640
designs per weight set, not two runs. F11's per-scaffold ceiling is redefined as
the maximum `|delta_q_density|` over cells that hit their target with
`final_temperature == 0.3`, and the F11 heatmap uses `final_temperature` from
the same run. This is a metric redefinition and was approved rather than assumed.

**F. Fold classes (Phase 1).** Precedence order `mainly_alpha`, `mainly_beta`,
`alpha_beta`, `loop_rich`, first match wins. A scaffold matching no class is
discarded and logged to `logs/fold_class_discards.csv`, never forced into a
class. DSSP 8-state to 3-state: `H`, `G`, `I` are helix; `E`, `B` are strand;
everything else is coil.

## Phase 1 execution, 2026-08-13

**Tools installed into the shared `py311`.** Only what `CLAUDE.md` authorises.

- `pip install tmtools` -> tmtools 0.3.0, for TM-align in Phase 6. Verified by
  running `tm_align` on identical coordinates and getting TM-score 1.0. numpy
  was already satisfied at 1.26.4 and did not change.
- DSSP: **not installed.** See the blocker below.

**DSSP has no pip distribution, so Section 2.1's instruction cannot be followed
as written.** The only PyPI candidate is `pydssp`, and it is not a substitute:
its source defines `C3_ALPHABET = ['-', 'H', 'E']` and `assign()` accepts only
`out_type` in `{'onehot', 'index', 'c3'}`. It never emits `G`, `I`, or `B`, so
the frozen Decision F mapping (`H, G, I -> helix`, `E, B -> strand`) cannot be
implemented on it, and its `H` is its own criterion rather than a merge of
DSSP's three helix codes. Using it would silently change fold-class assignment.
The `dssp` stage stops with a message instead. Real DSSP is available on conda
(conda-forge 4.6.1, salilab 3.0.0) but not pip, which Section 2.1 says is a
stop-and-ask. Awaiting a decision.

**DSSP install, 2026-08-13.** `conda install -c salilab dssp=3.0.0` into `py311`,
chosen over conda-forge because a dry-run solve showed conda-forge's dssp would
upgrade PyRosetta from 2024.19 to 2026.32. PyRosetta's version determines the
REU terms and FastRelax behaviour that Phase 5 measures, so that upgrade was
rejected.

Two things about this install need recording honestly.

First, it was not as self-contained as the dry-run appeared. The dry-run output
was read through a filter that only showed numpy, torch, pyrosetta, python and
dssp lines, so "nothing else changes" was an overstatement. The install actually
also brought in `libboost 1.84.0` and `libgcc 16.1.0` as new packages and
updated `ca-certificates`, `certifi`, `libgcc-ng` (13.2.0 to 16.1.0) and
`openssl`. numpy stayed at 1.26.4 and pyrosetta stayed at 2024.19, which were
the two that mattered. After the install, `numpy`, `scipy`, `pandas`,
`matplotlib`, `Bio`, `yaml`, `tmtools` and `torch` all still import and
`pyrosetta.init()` still succeeds, so the shared env is intact, but the
footprint was larger than reported to the user beforehand.

Second, the installed binary did not run:

```
mkdssp: error while loading shared libraries: libboost_thread.so.1.73.0
```

The salilab 3.0.0 build is linked against boost 1.73 while the solver installed
boost 1.84. Nothing else in `py311` used boost before this install, so the
version is free to move. Resolution is recorded below once settled.

`config/cluster.yaml` now carries an explicit `paths.dssp_bin` rather than
assuming the binary sits in the benchmark env, so DSSP can be moved to its own
env without touching any code.

**Bugs found and fixed in the Phase 1 code.**

1. `lib/io.py` shadowed the standard library's `io`, which the interpreter
   pre-imports, so `import io` returned the stdlib module and every `bio.*`
   reference raised `AttributeError`. `lib/` is now a package and scripts use
   `from lib import io as bio`. The filename stays as Section 3 specifies.
2. The exclusion list was read positionally. `excluded_PDBs.csv` is a pandas
   dump whose first column is an unnamed row index with the IDs under
   `PDB_IDS`, so `row[0]` built an exclusion set of integers and excluded
   nothing: 1120 of 1120 chains survived. Now read by column name with a hard
   error if the column is absent. Correct result: 17369 exclusion IDs, 1054
   candidates from 1120 chains.
3. Resolution was parsed by scanning `REMARK   2 RESOLUTION.` for the first
   float-parseable token, which is the remark number `2`. Every X-ray structure
   would have been recorded as 2.0 A and passed the 2.5 A cutoff. Now only the
   text after `RESOLUTION.` is scanned. Spot-checked: 1a2p is 1.50, 1a32 is 2.10.
4. Multi-model entries were counted once per model, giving NMR chain A of 1a90
   3348 residues instead of 108. Parsing now stops at `ENDMDL`. Rerunning filter
   with this fix changed no outcome, because multi-model entries are rejected by
   the experimental-method check first, but the residue counts were wrong.
5. `sbatch` copies the job script into a spool directory, so
   `dirname ${BASH_SOURCE[0]}` did not resolve back into the repo and job
   60552137 died on a missing `config/cluster.yaml`. The drivers now pass
   `--export=BM_ROOT=<Benchmark dir>` and the sbatch scripts fail loudly if it
   is absent or wrong.
6. `conda.sh` was recorded at `/projects/f_sdk94_1/conda/etc/profile.d/conda.sh`,
   which does not exist. That path is an envs directory, not a conda install.
   The base is `/projects/community/anaconda/2022.10/bd387`.

**Curation results.** 1120 CATH 4.2 test chains, 1054 after applying
`excluded_PDBs.csv`, 1054 structures fetched with 0 failures, 188 passed the
Section 4 filters, 179 cluster representatives at 30% identity and 80% coverage.

Rejection breakdown of the 866 filtered out: 481 multi-chain, 169 not X-ray,
105 resolution above 2.5 A, 68 outside 80 to 250 residues, 42 chain breaks
longer than 2, 1 missing backbone atoms.

Note on the multi-chain rejections, which are the largest single category: the
filter rejects any entry whose asymmetric unit contains more than one protein
chain, not merely entries where the target chain is part of an obligate
complex. This is the strict reading of Section 4 filter 3 and of the Phase 1
goal of monomeric scaffolds. It is recorded here because a looser reading would
have retained several hundred more candidates. It does not threaten the target
of 32, since 179 representatives remain.

The 179 representatives span 80 to 250 residues (median 135) with WT net charge
from -23 to +10 (median -3), 39 of them positive, and resolution 0.95 to 2.5 A.

## Phase 1 acceptance FAILURE, 2026-08-13

The curation pipeline ran to completion but **does not meet PLAN.md Section 4
acceptance**: 23 manifest rows instead of 33, and two fold classes under the
"populated with >= 6 usable scaffolds" bar.

| Fold class | Selected | Available in pool | q_wt > 0 |
|---|---|---|---|
| mainly_alpha | 8 | 20 | 5 |
| mainly_beta | 6 | 6 | 0 |
| alpha_beta | 8 | 142 | 32 |
| loop_rich | 0 | 0 | 0 |

Two independent causes, neither of which the executing agent may fix alone
because both change which scaffolds land in which class.

**Cause 1: the frozen precedence starves loop_rich.** Decision F fixed the order
as mainly_alpha, mainly_beta, alpha_beta, loop_rich with first match winning.
Exactly 9 scaffolds in the pool satisfy `loop_rich` (coil >= 0.50), and all 9 of
them also satisfy `alpha_beta` (helix >= 0.15 and strand >= 0.15), which is
checked first. So alpha_beta absorbs every one and loop_rich gets nothing.
Reordering to put loop_rich first moves those 9 across, giving loop_rich 9 (4
with positive q_wt) and leaving alpha_beta 133. Precedence alone cannot fix
mainly_beta, which has only 6 members under any ordering.

**Cause 2: the strict single-chain filter shrinks the pool.** The filter rejects
any entry whose asymmetric unit holds more than one protein chain, which removed
459 candidates. Re-testing those against every other Section 4 filter, using the
chain named in the split, shows **285 would be recovered** if the requirement
were instead that the designed chain is a single protein chain. That grows the
pre-clustering pool from 188 to 473. Beta-rich folds are disproportionately
oligomeric in the asymmetric unit, so this filter falls hardest on exactly the
class that is short.

The strict reading is defensible and arguably safer: a chain that is part of an
obligate complex has an interface that the LayerSelector will treat as surface,
and supercharging it would be measuring the wrong thing. The looser reading
gets the benchmark to its target of 8 per class. This is a scaffold-filter
definition, so it goes back to the plan owner rather than being decided here.

Nothing downstream has been run against the 23-row manifest.

## Phase 1 complete, 2026-08-13

Manifest: 25 scaffolds, 8 per fold class plus eGFP, meeting the amended
Section 4 acceptance. LayerSelector caches present for all 25 in `data/parsed/`
and all 7 control-arm scaffolds in `data/hbond_protected/parsed/`, no task
failures. `n_designable` filled for 25/25, range 47 to 96, median 63.

**A numbering bug in the repo script, found by a crashing task.**
`surface_selector` (protein_mpnn_supercharge.py:84) returns PDB numbering via
`pose.pdb_info().pose2pdb(res)`, but line 412 uses those values to index
`pose.sequence()`, which is pose numbering:

    indices = [i for i in indices if seq[i-1] not in ['G', 'P', 'C']]

The two agree only when the input is numbered exactly 1..N. Across this
manifest, 20 of 25 scaffolds have PDB numbering running past the sequence
length and raise IndexError, 2 are shifted but still in range and therefore
apply the Gly/Pro/Cys protection to the WRONG residues with no error at all
(4lws_B is numbered -2..85, 2gux_A is -4..121), and only 3 are numbered 1..N.

The silent case is the serious one: a wrong designable set, and every metric
downstream computed against it without any signal that something is off.

**eGFP is numbered 1..231, so the manuscript's published results are not
affected.** The bug only affects structures not numbered from 1, and all prior
work used eGFP. It would affect anyone applying the method to a PDB download,
which is worth a line in the Phase 9 README.

Fixed on the benchmark side, not in the repo script, which is not modified:
`01_prepare_structures.py:clean_structure` renumbers each cleaned scaffold to
1..N so PDB and pose numbering coincide. The crystal numbering is preserved in
`data/scaffolds/<id>/<id>_numbering.csv`.

**The per-arm cache separation is doing real work.** Comparing the two caches
for the 7 control-arm scaffolds, the designable sets differ in every case, for
example eGFP has 96 designable positions with `-mhbond` and 62 with h-bond
protection on. Had the arms shared one cache, as the repo script's cache key
implies they would, the control arm would have run against 96 and the h-bond
experiment would have measured nothing.

**The Decision 12 oligomer risk, measured rather than assumed.** Chains taken
from multi-chain asymmetric units could show inflated designable fractions,
because stripping the partner chains exposes an interface that LayerSelector
counts as surface. Across the 24 split-derived scaffolds the effect is not
detectable: Spearman rho between `n_chains_in_asu` and designable fraction is
0.086 (p = 0.69), and median designable fraction is 0.53 for monomeric versus
0.64 for oligomeric asymmetric units (Mann-Whitney p = 0.38). Individual
extremes exist, notably 3vjf_A at 0.95 designable from a 2-chain ASU. With
n = 24 this test is underpowered, so this is "no systematic shift detected at
this panel size", not "no effect". `n_chains_in_asu` is in `data/filtered.csv`
so Phase 7 can stratify on it.

## Phase 2 complete, 2026-08-13

456 of 456 cells succeeded, 4,560 designs written. Acceptance required >= 95%
of cells to produce a FASTA; the result is 100%. Every FASTA header parses
under `lib/io.py`, every design's header charge agrees with an independent
recomputation through `lib/charge.py`, and every native charge agrees with the
manifest `q_wt`.

Runtime 2.9 to 76.5 s per cell, median 9.2 s, on `main-redhat` with 4 CPUs.

Temperature behaviour under `-u`: 102 of 456 cells escalated above the 0.3
starting temperature, up to 19 escalations. Final temperature distribution is
0.3 for 354 cells, then 38, 25, 15, 10, 5 and 9 cells at 0.4 through 0.9. No
cell exceeded the 0.9 ceiling, so every escalating cell still terminated
normally. This matters for Decision E: F11's ceiling is defined as the cells
with `final_temperature == 0.3`, and with 102 cells escalating that column
carries real signal rather than being uniformly 0.3.

**A bug in the benchmark driver, exposed by Decision 12.** The first array run
failed 96 of 456 cells with `KeyError: 'seq_chain_A'`, all of them on the six
scaffolds whose chain is not A (`4cq4_B`, `3sjq_C`, `4lws_B`, `1li1_B`,
`4o32_C`, `3vto_Q`). `02_run_mpnn_supercharge.py` was not passing `--chain_id`,
whose default is `'A'`; `protein_mpnn_supercharge.py:759` builds
`designed_chain_list` from it, so `tied_featurize` looked for chain A in a
cleaned PDB holding only chain B, C or Q.

This was latent until Decision 12. Under the strict single-chain filter every
scaffold came from a single-chain asymmetric unit and was chain A, so the
default happened to be correct. Relaxing the filter admitted non-A chains and
broke the assumption. The driver now passes the chain from the manifest, and
`chain` is a column in `slurm/tasks_phase2.tsv`.

Re-emitting the task file after adding that column was verified not to disturb
task ordering: all 456 (task_id -> scaffold, method, target, seed) tuples are
bit-identical to the previous file, so the 96 failures were rerun by index with
their original seeds rather than repeating the 360 that had succeeded.

## Phase 3 complete, 2026-08-13

400 of 400 cells succeeded, jobs 60555884 (avnapsa) and 60555885 (rosetta). No
cell failed and none needed rerunning. Both arms cover the same 200 scaffold x
target grid as `mpnn_soluble`, exactly, and every cell wrote its FASTA, its PDBs
and its sidecar. That is Section 6 acceptance.

**PLAN.md Section 5's two determinism assumptions are both contradicted by the
finished arms.** They are recorded here rather than acted on, because changing
either is a schema change that goes through PLAN.md first.

*Rosetta is not near-deterministic.* `nstruct 10` exists to test that claim. Over
200 cells the median cell produced 7 distinct mutation sets out of 10 and the
mean is 6.6; only 3 cells produced a single set, and 35 cells produced 8. The
sampling spread is not incidental to the comparison: the arm hits its target on
23.2% of replicates but on at least one replicate in 70.5% of cells, so the
headline number depends entirely on whether the baseline is credited with one
draw or ten.

*AvNAPSA's determinism is still unmeasured, and this run cannot measure it.* The
arm ran at `nstruct 1` per the frozen matrix, so `n_unique_mutation_sets` is 1 in
all 200 cells by construction, not by observation. It should not be read as
confirming determinism. The reference runs in
`Former_Methods/eGFP/TargetPos2_Avn/` gave 65, 65 and 61 mutations across three
replicates, which points the other way. Measuring it costs about 12 CPU minutes
per extra replicate across the whole arm (median cell 16.5 s), so the question is
cheap to settle if PLAN.md Section 5 is amended to allow `nstruct > 1`.

**Where the two baselines actually miss is opposite.** Of 82 avnapsa misses, 73
overshoot the target and 9 undershoot. Of 1,536 rosetta misses, 976 undershoot
and 560 overshoot. Both degrade as |ΔQ| grows, avnapsa from 73% to 48% and
rosetta from 36% to 11% between the |ΔQ| 6-10 and 21+ buckets.

## Phase 4 built, 2026-08-13

`scripts/lib/phase4.py`, `scripts/05_run_vanilla_rejection.py`,
`scripts/06_run_random_control.py`, `slurm/array_phase4.sbatch`,
`slurm/tasks_phase4.tsv` (50 tasks: 25 rejection, 25 random control), plus a
`phase4` block in `benchmark.yaml` and `phase4_rejection` / `phase4_random` in
`cluster.yaml`.

Task granularity is the scaffold, not the cell. A rejection task draws one pool
of 2,000 samples that all 8 of that scaffold's targets are counted against, and a
random-control task walks its 8 targets in about 4 seconds, so a per-cell array
would have been 200 jobs of scheduling overhead for no rerun benefit.

**The fixed-positions conversion is the load-bearing part.** PLAN.md Section 7
requires the vanilla sampler to be restricted to exactly the designable set the
supercharging run uses, and that set exists only inside the Phase 1 pickle.
`lib/io.fixed_positions_for` reproduces the construction
`protein_mpnn_supercharge.py:703-706` performs, and both consumers go through it.
Three things were verified rather than assumed:

- The cached designable indices are 1-based positions into the cached wild-type
  sequence, and `lib/io.load_designable_cache` raises if any index falls outside
  `1..N`. That is the numbering bug from item 1 of the bug list, guarded at the
  point where it would now do damage.
- `protein_mpnn_run.py` keys its fixed-positions dict on the PDB basename
  (`protein_mpnn_utils.py:190`) and zeroes `mask[fixed - 1]`, so the 1-based
  convention matches on both sides.
- `--use_soluble_model` is passed instead of `--path_to_model_weights`. An
  explicit weights folder overrides the soluble/vanilla selection inside the
  runner, and the flag resolves to `soluble_model_weights/v_48_020.pt`, the same
  checkpoint file the primary arm loads.

After sampling, `run_rejection` checks that no sample differs from wild type at
any position outside the designable set. A violation fails the cell rather than
warning, because a pool that was not actually restricted is not comparable to the
supercharging runs and would silently understate the ablation.

**Three points where Section 7's one-sentence spec for the random control needed
resolving**, documented on `lib/phase4.random_charge_design`: positions already
carrying the target polarity are skipped (a K to R swap moves no charge); a pick
that would overshoot is skipped rather than accepted (reversing an Asp moves 2
units, a neutral residue moves 1, so with one unit left only a neutral position
lands the target exactly, and accepting the overshoot would have missed by one in
a large fraction of cells for reasons unrelated to the random prior); and Gly,
Pro and Cys need no exclusion because the cached designable set was built with
`mutate_glyprocys` False and already omits them, verified on `1a1x_A` and `eGFP`.

## Phase 4 complete, 2026-08-13

25/25 rejection pools (50,000 samples) and 200/200 random-control cells, 0
failures in either. `results/rejection_curve.csv` has 200 rows covering all 25
scaffolds at all 8 targets, and `results/rejection_distribution.csv` has 25.
Section 7 acceptance is met apart from threading the control designs, which is
Phase 5 work by PLAN.md's own ordering.

**Two results that change how the phase should be written up.**

*The coverage collapse is not gradual, it is a wall, and it is one-sided.* 119
of 200 cells produced zero on-target samples out of 2,000. At +16 and +24 ΔQ
density it is 50 of 50: across 50,000 vanilla samples spanning every scaffold and
every fold class, not one landed on a positive supercharging target. The cause is
a systematic negative bias in the base model rather than per-scaffold noise.
Vanilla soluble ProteinMPNN pulls every scaffold toward about -5 regardless of
where it started: `3sjq_C` has `q_wt` +14 and samples at -4.78, `4o32_C` has +10
and samples at -3.02, and only 2 of 25 scaffolds sample more positive than their
wild type. With sampling sd between 2.0 and 5.0, the positive rungs sit 6 to 22
sd out. No sample budget reaches them.

*The cost ratio is the weaker half of the argument and should not lead.* In the
81 cells where rejection found the target at all, it costs a median 8.6 CPU
seconds per on-target design against 0.62 s for supercharging, a median 15x with
a maximum of 834x. In the other 119 cells there is no ratio to quote, because
there is no hit to divide by; those rows are censored lower bounds. Reporting a
median speedup across all 200 cells would silently drop the 119 cells where the
alternative does not work at all, which is the stronger finding.

**The random control cannot be separated from the real method on any Phase 4
metric, by construction.** It reaches 100% of targets exactly, with 10 distinct
mutation sets in all 200 cells, and its mutation counts are identical to
`mpnn_soluble` to two decimal places (median 11.0, mean 13.25 for both). Charge,
hit rate and mutation load are all fixed by the arithmetic and the designable
set, which are held equal between the arms. Everything the control is for
therefore lands in Phase 5 and Phase 6. If those phases do not separate the two,
that is the finding, and it needs to be reported rather than explained away.

**PLAN.md amended.** Sections 0.1 and 6 now record the two Phase 3 decisions
(items 14 and 15) and the Decision B resolution (item 13); item B is struck from
Section 0.2 and item D is marked resolved with the measured timing. Section 6
keeps the superseded reimplementation requirement, marked, and flags its
`nstruct 10` diversity prediction as contradicted by the measurement. Standing
permission was given to edit PLAN.md without asking, so hard rule 1 no longer
gates it.

## Phase 5 and 6 bugs found while building, 2026-08-13

**11. `threading_only.py` does not relax a zero-mutation sequence, so the
"identical protocol" wild-type reference PLAN.md Section 8 asks for cannot come
from it.** Lines 234-239 short circuit when a sequence carries no mutations: the
pose is dumped and scored as-is, without FastRelax. Every design, by contrast, is
relaxed. Referencing relaxed designs against an unrelaxed crystal would put the
entire relaxation gain into `d_reu_per_res`.

Measured on eGFP, the tier A reference: crystal -555.27 REU, relaxed -795.38 REU,
**a gain of 240.11 REU over 231 residues**. Every eGFP design would have scored
about **1.04 REU per residue better than wild type from relaxation alone**, on a
metric whose real effects sit one to two orders of magnitude below that, and in
the flattering direction.

Fixed on the benchmark side in `lib/rosetta_metrics.relax_wt_reference`, which
uses the same score function (ref2015), the same `FastRelax(5)` repeats and the
same lowest-of-`num_relax` retention. `threading_only.py` is not modified.

**The fix is not literally "identical".** The design path restricts its movemap
and task factory to the mutated positions and their neighbour shell, and with
zero mutations that shell is empty, which is exactly the degenerate no-op above.
The reference therefore relaxes with an unrestricted movemap and repacks
everything. That gives the wild type more conformational freedom than any design
gets, which biases `d_reu_per_res` **against** the designs. It is the
conservative direction, but it is a bias and it is not neutral. Both the relaxed
score and the crystal score are recorded in every reference record
(`reu_total`, `reu_total_crystal`, `reu_relaxation_gain`) so the notebook can
show how much of any effect depends on this choice. Carry into Phase 9.

**12. ESMFold2's `--also-pdb` conversion zeroes the B-factor column.** The mmCIF
carries per-residue pLDDT in `_atom_site.B_iso_or_equiv`, but the biotite PDB
conversion written alongside it has 0.00 in every B-factor. Reading pLDDT from
the PDB returned a column of zeros that looked like a real measurement, and
`plddt_min_designable` would have been 0.0 for all 8,760 designs. Caught by
checking the first converted file rather than trusting it.
`lib/structure.read_plddt_per_residue` parses the CIF instead, and reads the
`_atom_site` loop header for column indices because the column order there is
not the conventional PDB-derived one (`B_iso_or_equiv` precedes `Cartn_x`). The
PDB is still used for the CA trace, where it is correct.

**13. Sharding a partial rerun on `SLURM_ARRAY_TASK_COUNT` silently folds
nothing.** `slurm/array_esmfold.sbatch` originally passed
`--num-shards "${SLURM_ARRAY_TASK_COUNT}"`. That is correct only for a full
array. Rerunning the 8 tasks that died on the bad GPU node with
`--array=2,28,29,30,31,32,33,39` set the task count to 8, so `fold_fasta.py`
re-split the FASTA into 8 shards and was asked for shards 28 through 39 of 8.
Those select no records. Every task exited 0 having folded nothing, and the array
looked like a clean success while 659 of 8,785 records were never attempted.
Caught by comparing the summary row count against the manifest rather than
trusting the exit codes. The shard count now comes from `BM_NUM_SHARDS`, exported
by the submitting driver, which is the same number for a rerun as for the
original submission.

**14. One GPU node had failing hardware.** All 8 failures in the first Phase 6
array landed on `gpu029` with `CUDA error: uncorrectable ECC error encountered`
at model load; all 31 tasks spread across the other 17 GPU nodes succeeded. Not a
code fault. `slurm.exclude_nodes` in `cluster.yaml` now carries the node list and
every submitter passes it through as `sbatch --exclude`, so a bad node is one
edit in the one file a new user is expected to touch. Remove `gpu029` from that
list once the node is repaired.

**15. ESMFold2 reports pLDDT on two different scales in two different files.**
The summary TSV's `plddt` column is 0 to 1 (`0.9232`); the mmCIF's
`_atom_site.B_iso_or_equiv` is 0 to 100 (`92.32`). `d_plddt_vs_wt_pred` was
computed as the design's CIF-derived mean minus the wild type's TSV-derived
value, so it subtracted a fraction from a percentage. The column averaged
**+73.55**, which is arithmetically impossible for a difference between two pLDDT
values and is how it was caught, rather than by reading the parser. Both values
now come from `lib/structure.read_plddt_per_residue`, so there is one scale and
one code path. Anything that reads that TSV's `plddt` directly must multiply by
100 first.

**16. Editing a library while an array is running kills the tasks that import it
mid-write.** Four Phase 5 tasks (60571755 indices 46 to 49) died with
`ImportError: cannot import name 'config' from 'lib'` and
`FileNotFoundError: .../lib/config.py`, in the seconds during which
`scripts/lib/config.py` was being rewritten to add `exclude_flag`. Not a defect
in the code: the tasks that started before and after it are fine, and rerunning
the four by index succeeds. Recorded because the failure mode looks like a
packaging bug and is not one. Do not edit `scripts/lib/` while an array is in
flight; wait for the drain, or stage the edit and submit afterwards.

**17. The repository root `.gitignore` would have silently dropped the notebook
and every design FASTA.** It excludes `*.ipynb`, `*.fa`, `*.fasta` and `*.pkl`
globally, rules that predate this benchmark and are reasonable for a repo whose
FASTAs are all generated output. Under them,
`Benchmark/notebooks/analysis.ipynb` and all 1,056 per-cell design FASTAs were
ignored, so a commit would have looked clean while omitting the artifact PLAN.md
Section 14 requires and the files that let a reader recheck any charge in
`designs.csv`. `Benchmark/.gitignore` now re-includes them by negation, which
works because those are file patterns rather than directory patterns. Verified
with `git add -An`: 2,709 files stage, including 1,056 FASTAs and the notebook,
while `threaded/`, `predictions/` and the 50,000-sequence rejection pools stay
excluded. The root file is not modified.

**18. Two AF3 output-parsing bugs, both of which produced a clean-looking CSV
with wrong contents.** Neither was an AF3 failure: all 36 jobs ran correctly.

*The mmCIF was handed to a fixed-column PDB parser.* `lib/structure.read_ca_trace`
slices by byte position, which is right for PDB and meaningless for mmCIF, where
fields are whitespace-delimited. `line[12:16]` lands mid-field, nothing matches
"CA", and the function returns an empty trace rather than raising. The first
collection reported 0 of 36 ok with "too few CA atoms to align: 0 vs 231", which
at least failed loudly; had TM-align tolerated an empty trace it would have
produced numbers. `read_ca_trace_cif` and a dispatching `read_ca_trace_any` now
exist, and the CIF reader parses the `_atom_site` loop header for column indices
rather than assuming an order.

*AF3 strips the '+' from the job name when it creates the output directory, as
well as lowercasing.* `eGFP_mpnn_soluble_q+30_s08` lands in
`egfp_mpnn_soluble_q30_s08`. Matching on `job_name.lower()` therefore resolved
every negative-charge job and missed every positive one: exactly 16 of 36, all of
them the `+16` density rows, silently recorded as "no model produced". A
benchmark that lost every positive-supercharging AF3 result while reporting the
negative ones would have been badly misleading, and the symmetry of the loss is
what made it obvious. `_resolve_job_dir` now tries the sanitiser's known
transformations and falls back to a normalised directory scan.

## The diversity claim needs a comparator, because random beats the method on it

Not a bug, a result, and one that changes how a sentence in the manuscript should
be worded. Medians per scaffold x target cell:

| metric | mpnn_soluble | rosetta | random_control |
|---|---|---|---|
| unique mutation sets (of 10) | 10 | 7 | 10 |
| mean pairwise Hamming | 9.2 | 2.8 | 18.3 |
| positional entropy (bits) | 0.28 | 0.09 | 0.64 |
| designable coverage | 0.41 | 0.19 | 0.70 |

**The random-charge control is more diverse than the method on every one of the
four measures**, by roughly a factor of two. That is the expected direction and
not a defect: the control picks positions uniformly, while the method
concentrates on the substitutions its prior likes, which is the entire point of
having a prior. But it means "the method produces diverse designs" is not a claim
that survives on its own. The defensible claim is the comparative one, and it is
strongly supported: against Rosetta the method gives 3 more unique mutation sets
per cell (Cliff's delta 0.86), 6.5 more pairwise Hamming distance (delta 0.94),
and roughly double the positional entropy (delta 0.97), all at
Holm-corrected p < 1e-28.

Pair this with the pLDDT result, where the ordering reverses: the method beats
the random control by 1.32 pLDDT. Together they say the prior trades diversity
for structural plausibility, which is a more interesting and more defensible
story than diversity alone.

## The AF3 subset is weak evidence, and its own controls say so

Not a bug. All 36 jobs ran and all 36 parsed, but the wild-type controls
PLAN.md Section 9 requires are what make the result interpretable, and they are
unflattering:

| scaffold | AF3 WT pLDDT | AF3 WT TM-score to its own crystal |
|---|---|---|
| eGFP | 30.7 | 0.343 |
| 3iu6_A | 48.0 | 0.343 |
| 1at0_A | 61.6 | 0.628 |
| 4o32_C | 89.0 | 0.927 |

**AlphaFold3 in single-sequence mode cannot fold three of these four wild-type
sequences.** A TM-score of 0.34 against the protein's own crystal structure is a
failed prediction, and eGFP, the focus scaffold, is the worst of them. Absolute
AF3 numbers for the designs therefore carry almost no information: a design
scoring 0.35 is indistinguishable from a wild type scoring 0.34.

This is the direct cost of the methodological choice Section 9 mandates, and the
choice is still right: building an MSA for a supercharged sequence retrieves
wild-type homologs and rescues the prediction with evolutionary signal the design
does not have. But the consequence has to be stated rather than absorbed. Read
the AF3 subset **only** as a design-versus-matched-WT comparison, never as an
absolute quality measure, and treat ESMFold, which folds these wild types well
(median TM 0.886), as the load-bearing structural evidence.

If a stronger AF3 result is wanted, the option that preserves the methodology is
to hold the **wild-type MSA fixed** for every sequence including the designs,
which Section 9 explicitly permits as an alternative to single-sequence mode.
That is a rerun of the 36 jobs and a change to `af3_msa_mode`, so it goes through
PLAN.md first. Not done.

## Phase 9 verification pass, 2026-08-14

Every PLAN.md Section 14 box was rechecked against files on disk rather than
against `PROGRESS.md`. Nine of nine hold. The evidence table is PLAN.md
Section 14.1. Four things were wrong and are fixed; two are recorded as open.

**The notebook had never been executed as a notebook.** All 26 code cells were
stored with `execution_count: null` and zero outputs, so the standing claim that
they "run clean" rested on `analysis_lib` having been called directly, not on a
top-to-bottom run. The cause is that `py311` has IPython but no jupyter,
nbconvert, nbformat or ipykernel, so the documented command could not run there:

```
$ jupyter nbconvert --to notebook --execute analysis.ipynb
timeout: failed to run command 'jupyter': No such file or directory
```

Both `README.md` and `run_all.sh` gave that command as the reproduction step, so
the one instruction a reader would follow to regenerate every figure failed
immediately. Installing jupyter into `py311` was not done: it is a shared lab
env and the standing rule is to install only DSSP and TM-align there.

Resolved two ways, both verified:

1. `/projects/f_sdk94_1/conda/envs/shared_als515` already has jupyter 5.8.1,
   nbconvert 7.16.6, ipykernel 6.30.1, numpy, pandas and matplotlib. It has no
   torch and no PyRosetta and so cannot run any other phase, but the notebook
   only reads CSVs. `jupyter nbconvert --to notebook --execute` completes in
   35 s, exit 0, no cell errors.
2. `scripts/run_notebook.py`, added here, runs the code cells in order in one
   namespace for envs with no jupyter. It refuses to run if any cell contains
   IPython magics or shell escapes, since those would need a real kernel; the
   notebook has none. 26/26 cells ok in 11 to 21 s under `py311`. Takes
   `--dry-run`.

**Cross-environment reproducibility, unplanned but worth recording.** Running
the notebook under both envs and diffing the outputs, all seven table CSVs are
byte-identical, across Python 3.11.8 with numpy 1.26.4 and pandas 2.2.1 versus
Python 3.13.5 with numpy 2.3.1 and pandas 2.3.3. No analysis number depends on
those versions. The figures and tables in the repository were written from
`py311`, and they are now written by the notebook run itself rather than by a
side script, so their provenance is the notebook.

**Notebook cells had no nbformat `id` fields.** `nbformat.validate` warns that
this becomes a hard error in future versions. Normalized; `normalize()` reported
0 other changes and the notebook still runs clean afterwards. It is still stored
without cell outputs.

**Two stale sections in `PROGRESS.md`.** Its "Not done" list said the result
CSVs, notebook, figures and tables did not exist, and a later heading said
Phase 5 was "Not started". Both contradicted the status table in the same file
and both are corrected. Anyone picking the work up cold would have been told to
rebuild finished phases.

## Instructions taken 2026-08-26

**AvNAPSA `nstruct` raised from 1 to 10.** Author's instruction: sample the same
number of sequences as the MPNN and Rosetta arms. Recorded as PLAN.md Section
0.1 item 16, Section 5's sample table amended, Section 0.2's open item closed,
`config/benchmark.yaml` updated. `slurm/tasks_phase3.tsv` regenerated: the diff
against the `nstruct 1` file touches the 200 avnapsa rows in the `nstruct`
column only, seeds and every other field unchanged, and the 200 rosetta rows
are untouched. The `nstruct 1` outputs are preserved under
`logs/nstruct1_backup/` (200 sidecars, 200 FASTAs, the old task file) so the
`s00` reproduction check can be made against them. Replicates run in one
PyRosetta RNG stream per cell from the same `-jran`, exactly as the rosetta arm
does, so `s00` is expected to reproduce the `nstruct 1` sequence bit for bit.

**`-mhbond` is no longer treated as an error to be corrected.** Author's
decision: the flag stays as it is and its documentation stands. The claims that
its polarity contradicts the manuscript Methods and needs correcting are removed
from `MANUSCRIPT_CHECKLIST.md`, `PROGRESS.md`, `README.md`, this file's
`## Discrepancies` section and PLAN.md Sections 5 and 13. The semantics are
still recorded, because `hbond_filter` is the inverse of the flag and cannot be
read without them. Nothing about how the arms were run changes: the primary arm
passed `-mhbond` before and passes it now.

**The Amarel CPU and GPU partitions were renamed cluster-side.** `main-redhat`
and `gpu-redhat` no longer resolve; `sbatch` rejects them with "invalid
partition specified: main-redhat". `sinfo` on 2026-08-26 lists `main` (509
nodes), `gpu` (55) and `p_sdk94_1` (1). This broke every submission path until
`config/cluster.yaml` was updated, which was the only file needing the change.
Node families are unchanged: phase 3 ran on `hal0xx` and `halk0xx` under the old
name and the 2026-08-26 timing cell landed on `hal0341` under the new one, so
this is a rename rather than new hardware, and Table T5's requirement that
runtimes be compared across identical CPU allocations still holds at 4 CPUs.
Rows written before 2026-08-26 carry `partition="main-redhat"` and rows written
after carry `partition="main"`; both refer to the same partition.

**`py311` now has jupyter, nbconvert, nbformat and ipykernel.** Installed by the
author, not by this benchmark. The no-jupyter fallback `scripts/run_notebook.py`
is kept, since it is what the Phase 9 cross-environment check was run with and
it costs nothing, but the nbconvert route now works from `py311` directly and no
longer needs `shared_als515`.

**15 of 199 avnapsa tasks failed on transient NFS reads of the shared conda
env, not on anything in the benchmark.** Job 60979987 launched 199 tasks at
once; 15 died at 55 to 58 s with import errors inside `site-packages/yaml`:
`ModuleNotFoundError: No module named 'yaml.constructor'` on some,
`FileNotFoundError: .../yaml/emitter.py` on others, for files that are present
and that the other 184 tasks imported without trouble. This is `/projects`
contention under a simultaneous 199-task start, not a code or data problem.
Rerun of the 15 indices as job 60981048. Recorded because it will recur on any
array this wide and the fix is to rerun by index, not to change anything.


## Instructions taken 2026-09-05

Three new arms, nine reworked figures, one new figure, and a second notebook.
PLAN.md Section 0.1 items 17 to 21 record the decisions; this file records what
had to be worked around to implement them.

### The repo-root supercharge script cannot load an arbitrary checkpoint

`protein_mpnn_supercharge.py:618-622` builds its checkpoint path as
`<path_to_weights>/{vanilla,soluble}_model_weights/<model>.pt`. There is no
`--checkpoint` flag, and the if/elif on `--weights` has no `else`, so any value
other than `original` or `soluble` falls through and raises `UnboundLocalError`
at line 622 with no message about the real cause. Line 625 then appends the same
`--path_to_weights` to `sys.path` to import `protein_mpnn_utils`, so that one
argument is doing two jobs.

Not worked around by patching the script: it is published and referenced by the
manuscript. Instead each biased arm points `--path_to_weights` at a shim
directory under `Benchmark/`:

    data/altweights/hyper/protein_mpnn_utils.py            -> the patched clone's copy
    data/altweights/hyper/vanilla_model_weights/v_48_020.pt -> HyperMPNN/v48_020_epoch300_hyper.pt
    data/altweights/halo/...                                -> HaloMPNN/epoch_last.pt

Both are symlinks, which is worth flagging because an earlier entry in this file
says the arm working directories are not symlinked. That statement was about the
LayerSelector cache, where a symlink would have silently shared a designable set
between arms. These symlinks are read-only pointers to immutable checkpoints and
share nothing between arms.

Verified before running the arrays rather than assumed:

  * Both checkpoints load and are architecturally identical to
    `soluble_model_weights/v_48_020.pt`: 118 state-dict tensors, matching key
    names, `num_edges` 48, `noise_level` 0.2. The extra `epoch`, `step` and
    `optimizer_state_dict` keys are ignored by the loader at
    `protein_mpnn_supercharge.py:766-771`.
  * The two symlinks resolve to different files (md5 28174547... and
    1e8379f2...), despite both checkpoints being 20,067,862 bytes.
  * One test cell each, `1a1x_A` at the -32 rung, produced three different
    sequences for `mpnn_soluble`, `mpnn_hyper` and `mpnn_halo`. Identical
    sequences would have meant the shim fell back to the stock weights, which is
    the failure this check exists to catch.

Consequence for the results: `--weights` reads `original` on the command line
for both biased arms, because the shim reuses the `vanilla_model_weights`
directory name. Left as-is, the `weights` column of `designs.csv` would have
been indistinguishable from `mpnn_vanilla_weights`. The column now records the
shim directory name, `hyper` or `halo`, so it names the checkpoint that actually
produced the row. `02_run_mpnn_supercharge.py:weights_label` is the one place
that mapping happens.

### `rosetta` and `mpnn_soluble` were not matched on h-bond handling

`scripts/lib/baseline.py` hardcoded `dont_mutate_hbonded_sidechains(True)`, so
the `rosetta` arm protected h-bonded sidechains while the primary MPNN arm,
which passes `-mhbond`, did not. The hydrogen-bond figure compared the two
anyway. That is not a bug in either arm, but it means the difference that figure
reported carried a confound that was never stated. (That figure is F9 in
`analysis.ipynb` and F6 in `analysis_updated.ipynb`; see README.md for the
mapping between the two numbering schemes.)

Fixed by making it a per-arm config value defaulting to `True`, so `rosetta` is
byte-unchanged, and adding `rosetta_hbond_off` with it set `False`. The new arm
runs on the same 7-scaffold control subset as `mpnn_soluble_hbond_protected`.
To guarantee the two subsets are identical rather than coincidentally equal, the
selection rule moved out of `02_run_mpnn_supercharge.py` into
`lib/io.control_subset` and `lib/baseline.control_subset_per_class` reads the
count off the MPNN control arm's config rather than carrying its own copy.

`hbond_filter` was null for every baseline row. It now carries the real value on
the two score-based arms and stays null on `avnapsa`, whose sequence-based
surface definition never consults the switch. Writing `True` there would have
claimed a setting the arm did not use.

### `hbond_filter` was null on 200 completed `rosetta` cells

Adding `rosetta_hbond_off` made `hbond_filter` meaningful on the score-based
arms, but the 200 `rosetta` cells had already run and written null, because the
setting was hardcoded and there was nothing per-arm to record. A figure that
compares protection on against protection off cannot read a null on the "on"
side.

Not fixed by rerunning the arm, which would cost 16 CPU-hours to recompute
sequences that are already correct, and not fixed by editing the sidecars, which
are the recorded outputs of a completed run. The value was never unknown: every
one of those sidecars carries `mover_settings.dont_mutate_hbonded_sidechains`,
which is what the mover was actually configured with, and it reads True in all
200. `10_aggregate.hbond_filter_for` reads it from there when the direct field
is absent. Nothing on disk was rewritten and no value was assumed.

AvNAPSA stays null on purpose. Its sequence-based surface definition never
consults that switch, so there is no setting to report, and writing True or
False would claim one the arm did not use.

Checked across all 1,512 cells after the change: `mpnn_soluble`,
`mpnn_vanilla_weights`, `mpnn_hyper`, `mpnn_halo` all False;
`mpnn_soluble_hbond_protected` and `rosetta` all True; `rosetta_hbond_off` all
False; `avnapsa` and `random_control` blank.

### `lib/io.parse_fasta` cannot read the Phase 4a pools

First attempt at `scripts/12_rejection_hit_metrics.py` used `bio.parse_fasta`
and failed on every pool with `header '1a1x_A' has no charge`. The pools are
written by the stock `protein_mpnn_run.py`, whose header is
`>T=0.3, sample=1, score=..., seq_recovery=...` with no charge field, while
`lib/io.parse_fasta` parses the supercharge script's `key=value` headers and
requires `charge=`. `lib/phase4.parse_vanilla_fasta` already exists for exactly
this and additionally checks the pool's native record against the cached wild
type, so the script uses it instead of growing a third parser.

### Task-id stability

Both new MPNN arms were appended to the end of `arms:` and `rosetta_hbond_off`
to the end of `baseline_arms:`, because `--emit-tasks` numbers `task_id` in arm
order. Checked rather than trusted: after re-emitting, the first 456 rows of
`tasks_phase2.tsv` and the first 400 of `tasks_phase3.tsv` are unchanged on
every shared column, so `--array=` reruns of the completed arms still address
the same cells. `tasks_phase2.tsv` gained a `weights_root` column, empty on
every pre-existing row.

## Open

**AvNAPSA `nstruct`. CLOSED 2026-08-26.** Raised to 10 on the author's
instruction and the 200-cell arm rerun; see `## Instructions taken 2026-08-26`
above. 200 of 200 cells ok, 4.45 CPU-hours, and all 200 reproduce their
`nstruct 1` sequence at `s00`. Measured `n_unique_mutation_sets` out of 10:
150 cells return 1, 39 return 2, 8 return 3, 1 returns 4, 2 return 5. Median 1,
mean 1.33. The metric is now an observation and may be reported.

**`fold_class` is null on the 408 eGFP rows of `designs.csv`.** eGFP enters as
the focus scaffold, not through CATH curation, so it was never DSSP-classified
and its manifest row carries `is_focus=True`, `source_split=focus_egfp` and
empty `fold_class`, `resolution` and `frac_*`. Fold-class figures therefore
describe the 24 curated scaffolds, with eGFP in the eGFP-specific panels.
Filling it in would mean either running eGFP through the Section 4 classifier,
which changes what F1 and T1 report, or writing a class by hand, which rule 4
forbids. Recorded, not resolved. PLAN.md Section 14.2 item 1.

**`logs/mpnn/` would be committed: 456 files, 21 MB, 58% of the 36 MB the first
commit would carry.** These are per-cell ProteinMPNN stdout logs, regenerable
output of the same kind as `logs/slurm/`, which `.gitignore` excludes. Section
14 does not name `logs/mpnn/`, so the checklist passes as written, and
`results/cells/*.json` already carries the per-cell provenance independently.
Excluding it is a one-line `.gitignore` change but it changes what is published,
so it is the author's call. PLAN.md Section 14.2 item 2.

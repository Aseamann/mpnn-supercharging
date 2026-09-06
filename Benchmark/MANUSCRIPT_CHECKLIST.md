# Manuscript and repository checklist

PLAN.md Section 13 asks for these to be handed back as a checklist rather than
edited blind. Nothing here has been applied: the manuscript and the repository
root are outside `Benchmark/`, and editing the manuscript is out of scope for
this work.

## Typographical and structural, from PLAN.md Section 13

- [ ] "Compairson" in the Results section heading
- [ ] "furtherst"
- [ ] "BLOSSUM62" should be BLOSUM62
- [ ] "canidate"
- [ ] "neutrual" (Figure 1 caption)
- [ ] "Institutue" (affiliation 1)
- [ ] "Despite these success"
- [ ] "were automatically adjust"
- [ ] Results sections are numbered 2.1 and 2.2 under a heading numbered 3, and
      Conclusion is numbered 4 with no section 3
- [ ] Author list reads "Antonio DeChillis" while the reference list uses
      "DeChellis A" throughout

## Repository, from PLAN.md Section 13

- [ ] The stray "not-excluded" typo in Design Notes.
- [ ] Add the benchmark outputs to the data availability statement.
- [x] `Benchmark/README.md` with a description and the `run_all.sh` invocation.

## Corrections the benchmark forces, not in PLAN.md Section 13

These come out of the results and are the ones that change claims rather than
spelling. Each is documented in `logs/ISSUES.md`.

- [ ] **Add a numbering caveat to the README.** `surface_selector` returns PDB
      numbering (line 84) and then indexes `pose.sequence()` with it (line 412).
      Those agree only when the input is numbered 1..N. Twenty of 25 raw CATH
      scaffolds crashed and two more silently applied the Gly/Pro/Cys protection
      to the wrong residues. eGFP is numbered 1..231, so the published results
      are unaffected, but anyone applying the method to a fresh PDB download hits
      this immediately. The benchmark works around it by renumbering; the repo
      script is unmodified.

- [ ] **The manuscript's diversity claim about Rosetta needs restating from the
      measurement.** PLAN.md Section 6 predicted `nstruct 10` would return
      "near-identical mutation sets". Over 200 cells the Rosetta arm returned a
      median of **7 distinct mutation sets out of 10** (mean 6.6; only 3 of 200
      cells returned a single set). The MPNN half of the claim does hold: 192 of
      200 cells return all 10 distinct.

- [ ] **AvNAPSA's near-determinism is now measured and can be stated.** The arm
      was rerun at `nstruct 10` on 2026-08-26, matching every other sampled arm
      (PLAN.md Section 0.1 item 16). Over 200 cells it returns a median of 1
      distinct mutation set out of 10, mean 1.33, with 150 of 200 cells
      returning a single set and a maximum of 5. Against Rosetta's median of 7
      and `mpnn_soluble`'s 192 of 200 cells returning all 10 distinct, the
      contrast is real and is an observation rather than a construction. The
      earlier caution that it must not be reported as 1 no longer applies. Note
      for the response letter: the reference runs in
      `Former_Methods/eGFP/TargetPos2_Avn/` gave 65, 65 and 61 mutations across
      three replicates, which does not match a fully deterministic protocol, so
      the wording should be near-deterministic and not deterministic.

- [ ] **Sequence selection was performed in t-SNE space and should be restated.**
      Reviewer 1 is right that those distances are not meaningful. The corrected
      procedure selects the medoid and the divergent representative in the
      original BLOSUM62 distance space and uses classical MDS for visualisation
      only (`scripts/lib/selection.py`, notebook section 11). Whether the eGFP
      selection changes cannot be checked from this repository: no record of the
      original selection exists here. The corrected indices are reported in the
      notebook so the comparison can be made against the manuscript by hand.

- [ ] **State the ΔREU reference protocol explicitly in Methods.** All ΔREU
      values are against a relaxed wild type built by this benchmark, because
      `threading_only.py` does not relax a zero-mutation sequence. Referencing
      against the unrelaxed crystal would have folded a median 305 REU of
      relaxation gain into every delta. The reference relaxes unrestricted while
      designs relax within the shell around their mutations, which biases ΔREU
      against the designs; that is the conservative direction but it is a bias
      and belongs in Methods.

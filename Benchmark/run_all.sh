#!/bin/bash
# Ordered list of every phase of the benchmark.
#
# This is NOT an unattended driver and deliberately does not chain phases with
# job dependencies. Each phase submits a Slurm array that must drain before the
# next starts, several require a timing task to be inspected before the full
# array is sized, and two require a human decision recorded in PLAN.md. Run it
# phase by phase, or with --dry-run to see every command without submitting.
#
#   bash run_all.sh --dry-run
#
# Failed array cells rerun by index. Seeds derive from (scaffold, method,
# target), so rerunning one index reproduces its original run:
#
#   python scripts/02_run_mpnn_supercharge.py --submit --array=17,43,91
#
# Watch the queue with:
#   squeue -u "$USER" -o "%.10i %.14j %.9P %.8T %.10M %R"
#   squeue -u "$USER" -h -t pending,running -r | wc -l   # against the 500 cap

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY="--dry-run"
    echo "### DRY RUN: nothing is submitted or written ###"
fi

run () { echo; echo "\$ $*"; [[ -n "${DRY}" ]] || "$@"; }

echo "=== Phase 1: scaffold curation and the LayerSelector cache ==="
run python scripts/00_curate_scaffolds.py --stage all ${DRY}
run python scripts/01_prepare_structures.py --emit-tasks ${DRY}
run python scripts/01_prepare_structures.py --submit ${DRY}

echo
echo "=== Phase 2: the charge ladder ==="
run python scripts/02_run_mpnn_supercharge.py --emit-tasks ${DRY}
run python scripts/02_run_mpnn_supercharge.py --time-one 0 ${DRY}
echo "# inspect wall_seconds in results/cells/, set slurm.phase2_mpnn.time, then:"
run python scripts/02_run_mpnn_supercharge.py --submit ${DRY}

echo
echo "=== Phase 3: classical baselines ==="
run python scripts/03_run_avnapsa.py --emit-tasks ${DRY}
run python scripts/03_run_avnapsa.py --submit ${DRY}
run python scripts/04_run_rosetta_supercharge.py --submit ${DRY}

echo
echo "=== Phase 4: rejection ablation and random control ==="
run python scripts/05_run_vanilla_rejection.py --emit-tasks ${DRY}
run python scripts/05_run_vanilla_rejection.py --time-one 24 ${DRY}
run python scripts/05_run_vanilla_rejection.py --submit ${DRY}
run python scripts/06_run_random_control.py --submit ${DRY}
echo "# after the pools land:"
run python scripts/05_run_vanilla_rejection.py --curve ${DRY}

echo
echo "=== Phase 5: threading and scoring ==="
echo "# the wild-type references MUST drain before the design tasks: a design"
echo "# task with no reference fails rather than scoring against a crystal pose"
run python scripts/07_thread_and_score.py --emit-tasks ${DRY}
run python scripts/07_thread_and_score.py --submit --kind wt ${DRY}
run python scripts/07_thread_and_score.py --submit --kind design ${DRY}
echo "# NOTE: the design block exceeds the 500-job submit cap and must be"
echo "# chunked, e.g. --array=25-474 then --array=475-576"

echo
echo "=== Phase 6: structure prediction ==="
run python scripts/08_predict_esmfold.py --emit-fasta ${DRY}
run python scripts/08_predict_esmfold.py --submit --shards 40 ${DRY}
run python scripts/08_predict_esmfold.py --collect ${DRY}
echo "# AF3 picks its scaffolds from the ESMFold wild-type pLDDTs, so it comes after"
run python scripts/09_predict_af3_subset.py --select ${DRY}
run python scripts/09_predict_af3_subset.py --submit ${DRY}
run python scripts/09_predict_af3_subset.py --collect ${DRY}

echo
echo "=== Phase 7: aggregation and statistics ==="
run python scripts/10_aggregate.py ${DRY}
run python scripts/11_statistics.py ${DRY}
run python scripts/capture_environment.py ${DRY}

echo
echo "=== Phase 8: notebook, figures, tables ==="
echo "# needs no cluster access; runs from results/*.csv alone, about 20 s"
echo "# py311 has no jupyter, so the bundled runner is the route from here."
echo "# The nbconvert route is equivalent and runs from shared_als515:"
echo "#   conda activate /projects/f_sdk94_1/conda/envs/shared_als515"
echo "#   cd notebooks && jupyter nbconvert --to notebook --execute analysis.ipynb"
run python scripts/run_notebook.py ${DRY}

echo
echo "All phases listed. Check logs/ISSUES.md before trusting any number."

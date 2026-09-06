#!/bin/bash
#SBATCH --job-name=bm_curate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 1 curation pipeline: fetch, filter, cluster. A single serial job, not an
# array, because each stage consumes the whole output of the previous one.
#
# The pool stage is deliberately not run here. It is a single small download and
# was run interactively so the split artifact could be checked before committing
# a job to it.
#
# This job stops before the dssp stage. That stage needs a real DSSP binary,
# which is not installed, and pydssp is not a substitute because it emits only
# three states. See logs/ISSUES.md.
#
# Submit with:  sbatch slurm/phase1_curate.sbatch   (resources passed on the CLI)

set -euo pipefail

# sbatch copies this script into a spool directory, so BASH_SOURCE does not
# point back into the repo. The submitting driver passes the root explicitly
# with --export=BM_ROOT=..., which also keeps the job env clean of the
# submitting shell.
BENCHMARK_ROOT="${BM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${BENCHMARK_ROOT}" || ! -f "${BENCHMARK_ROOT}/config/cluster.yaml" ]]; then
    echo "BM_ROOT must point at the Benchmark directory (got '${BENCHMARK_ROOT}')" >&2
    exit 1
fi
cd "${BENCHMARK_ROOT}"

CONDA_SH=$(python3 -c "import yaml;print(yaml.safe_load(open('config/cluster.yaml'))['conda']['conda_sh'])")
BENCH_ENV=$(python3 -c "import yaml;print(yaml.safe_load(open('config/cluster.yaml'))['conda']['benchmark_env'])")

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${BENCH_ENV}"

echo "host=$(hostname) job=${SLURM_JOB_ID}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

# The fetch stage needs outbound HTTPS. Fail immediately with a clear message
# rather than after an hour of silently failed downloads.
python - <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen("https://files.rcsb.org/download/1UBQ.pdb", timeout=30).read(64)
    print("RCSB reachable from this node")
except Exception as exc:
    sys.exit(f"no outbound HTTPS to RCSB from this node: {exc}")
PY

echo "=== fetch ==="
python scripts/00_curate_scaffolds.py --stage fetch

echo "=== filter ==="
python scripts/00_curate_scaffolds.py --stage filter

echo "=== cluster ==="
python scripts/00_curate_scaffolds.py --stage cluster

echo "=== done: fetch, filter, cluster complete ==="
echo "next blocked stage is dssp; see logs/ISSUES.md"

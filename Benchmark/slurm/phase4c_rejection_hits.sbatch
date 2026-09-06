#!/bin/bash
#SBATCH --job-name=bm_phase4c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 4c. One job, no array: score the on-target samples already sitting in
# the Phase 4a rejection pools so F4 can carry an unguided-MPNN series.
#
# Submit with:
#   python scripts/12_rejection_hit_metrics.py --submit

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

# Table T5 compares runtime across methods, which is only meaningful if every
# method gets the same CPU allocation. Pin thread counts so a task cannot
# quietly grab more cores than it was allocated.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host=$(hostname) job=${SLURM_JOB_ID}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

python scripts/12_rejection_hit_metrics.py --run

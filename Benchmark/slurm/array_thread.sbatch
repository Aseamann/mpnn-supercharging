#!/bin/bash
#SBATCH --job-name=bm_thread
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 5 threading and scoring. One array task is either one wild-type
# reference (scaffold x tier) or one design cell.
#
# The two kinds share slurm/tasks_phase5.tsv and this script, but are submitted
# separately and in order: every design task needs its tier's wild-type
# reference on disk before it can compute a delta.
#
#   python scripts/07_thread_and_score.py --submit --kind wt
#   # wait for that to drain
#   python scripts/07_thread_and_score.py --submit --kind design
# Rerun failures:
#   python scripts/07_thread_and_score.py --submit --array=104,220

set -euo pipefail

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

# threading_only.py fans out with a multiprocessing pool sized by --workers,
# which this driver sets from SLURM_CPUS_PER_TASK. Pin the per-process thread
# counts to 1 so the pool does not oversubscribe the allocation: N workers each
# spawning N BLAS threads is what turns a 4-CPU job into a 16-way thrash.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

python scripts/07_thread_and_score.py --task "${SLURM_ARRAY_TASK_ID}"

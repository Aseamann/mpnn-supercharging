#!/bin/bash
#SBATCH --job-name=bm_phase4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 4, one array task per scaffold per experiment.
#
# Both experiments share slurm/tasks_phase4.tsv, so this script dispatches on
# the task file's `kind` column rather than needing one array per experiment.
# They are submitted separately because their resource requests differ: a
# rejection pool is 2,000 forward passes, a random-control task is a fraction of
# a second of pure Python.
#
# Submit with:
#   python scripts/05_run_vanilla_rejection.py --time-one    # size the array first
#   python scripts/05_run_vanilla_rejection.py --submit
#   python scripts/06_run_random_control.py --submit
# Rerun failed tasks:
#   python scripts/05_run_vanilla_rejection.py --submit --array=3,7

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

TASK_FILE="slurm/tasks_phase4.tsv"
if [[ ! -f "${TASK_FILE}" ]]; then
    echo "missing ${TASK_FILE}; run --emit-tasks first" >&2
    exit 1
fi

# Column 2 is `kind`. Skip the header row.
KIND=$(awk -v id="${SLURM_ARRAY_TASK_ID}" -F'\t' 'NR>1 && $1==id {print $2; exit}' "${TASK_FILE}")
if [[ -z "${KIND}" ]]; then
    echo "no row with task_id=${SLURM_ARRAY_TASK_ID} in ${TASK_FILE}" >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} kind=${KIND}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

case "${KIND}" in
    rejection)      python scripts/05_run_vanilla_rejection.py --task "${SLURM_ARRAY_TASK_ID}" ;;
    random_control) python scripts/06_run_random_control.py    --task "${SLURM_ARRAY_TASK_ID}" ;;
    *) echo "unknown kind '${KIND}' for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

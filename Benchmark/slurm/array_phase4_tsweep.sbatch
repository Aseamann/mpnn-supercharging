#!/bin/bash
#SBATCH --job-name=bm_p4tsw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 4a temperature sweep, one array task per scaffold per temperature.
#
# PLAN.md Section 0.1 item 23. A separate task file and a separate script rather
# than more rows in tasks_phase4.tsv, so the T = 0.3 array indices stay valid for
# reruns and a stray --array here can never land on a T = 0.3 output path.
#
# Every row of tasks_phase4_tsweep.tsv is kind=rejection, so unlike
# array_phase4.sbatch this does not dispatch on the kind column. The temperature
# comes from the task row inside the driver, not from a flag here, so the file
# on disk is the single source of truth for what temperature a task ran at.
#
# Submit with:
#   python scripts/05_run_vanilla_rejection.py --emit-tasks --sweep --temperatures 0.6,0.9
#   python scripts/05_run_vanilla_rejection.py --time-one 24 --sweep   # size it first
#   python scripts/05_run_vanilla_rejection.py --submit --sweep
# Rerun failed tasks:
#   python scripts/05_run_vanilla_rejection.py --submit --sweep --array=3,7

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
# quietly grab more cores than it was allocated. The sweep pools are compared
# against the T = 0.3 pools on seconds per sample, so this has to match
# array_phase4.sbatch exactly.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_FILE="slurm/tasks_phase4_tsweep.tsv"
if [[ ! -f "${TASK_FILE}" ]]; then
    echo "missing ${TASK_FILE}; run --emit-tasks --sweep first" >&2
    exit 1
fi

# Columns: 1 task_id, 2 kind, 3 scaffold_id, ..., 10 temperature. Skip the header.
read -r KIND SCAFFOLD TEMP < <(awk -v id="${SLURM_ARRAY_TASK_ID}" -F'\t' \
    'NR>1 && $1==id {print $2, $3, $10; exit}' "${TASK_FILE}")
if [[ -z "${KIND:-}" ]]; then
    echo "no row with task_id=${SLURM_ARRAY_TASK_ID} in ${TASK_FILE}" >&2
    exit 1
fi
if [[ "${KIND}" != "rejection" ]]; then
    echo "task ${SLURM_ARRAY_TASK_ID} has kind '${KIND}', expected rejection" >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "scaffold=${SCAFFOLD} temperature=${TEMP}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

python scripts/05_run_vanilla_rejection.py --task "${SLURM_ARRAY_TASK_ID}" --sweep

#!/bin/bash
#SBATCH --job-name=bm_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 3 classical baselines, one array task per (method, scaffold, target).
#
# Both arms share slurm/tasks_phase3.tsv, so this script dispatches on the
# task file's `method` column rather than needing one array per arm.
#
# Submit with:
#   python scripts/03_run_avnapsa.py --submit
#   python scripts/04_run_rosetta_supercharge.py --submit
# Rerun failed cells:
#   python scripts/04_run_rosetta_supercharge.py --submit --array=17,43,91

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

TASK_FILE="slurm/tasks_phase3.tsv"
if [[ ! -f "${TASK_FILE}" ]]; then
    echo "missing ${TASK_FILE}; run --emit-tasks first" >&2
    exit 1
fi

# Column 4 is `method`. Skip the header row.
METHOD=$(awk -v id="${SLURM_ARRAY_TASK_ID}" -F'\t' 'NR>1 && $1==id {print $4; exit}' "${TASK_FILE}")
if [[ -z "${METHOD}" ]]; then
    echo "no row with task_id=${SLURM_ARRAY_TASK_ID} in ${TASK_FILE}" >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} method=${METHOD}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

case "${METHOD}" in
    avnapsa) python scripts/03_run_avnapsa.py --task "${SLURM_ARRAY_TASK_ID}" ;;
    rosetta|rosetta_hbond_off)
        python scripts/04_run_rosetta_supercharge.py \
            --task "${SLURM_ARRAY_TASK_ID}" --method "${METHOD}" ;;
    *) echo "unknown method '${METHOD}' for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

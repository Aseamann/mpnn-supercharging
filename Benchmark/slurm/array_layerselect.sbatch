#!/bin/bash
#SBATCH --job-name=bm_layersel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 1 LayerSelector cache, one array task per (scaffold, arm working dir).
#
# Partition, cpus-per-task, mem, time, array spec, and the log paths are all
# passed on the sbatch command line by 01_prepare_structures.py --submit, which
# reads them from config/cluster.yaml. Nothing cluster-specific is hardcoded
# here, so a new user edits cluster.yaml and nothing else.
#
# Submit with:  python scripts/01_prepare_structures.py --submit
# Rerun failed: python scripts/01_prepare_structures.py --submit --array=17,43,91

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

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$(which python)"

python scripts/01_prepare_structures.py --task "${SLURM_ARRAY_TASK_ID}"

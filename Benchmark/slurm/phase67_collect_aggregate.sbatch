#!/bin/bash
#SBATCH --job-name=bm_collect_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 6 collect plus Phase 7 aggregation and statistics, in one job.
#
# run_all.sh runs these four steps inline. They are put in a job here because
# CLAUDE.md forbids running anything longer than about a minute in the
# orchestrating shell, and collecting 10,585 predictions is longer than that.
# Order matters: --collect writes results/esmfold.csv, 10_aggregate.py reads it
# into designs.csv, and 11_statistics.py reads designs.csv.
#
# Submit with:
#   sbatch --export=BM_ROOT=$PWD --partition=main --cpus-per-task=4 --mem=32G \
#          --time=02:00:00 --output=logs/slurm/phase7/%j.out \
#          slurm/phase67_collect_aggregate.sbatch

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host=$(hostname) job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"

echo "--- 08_predict_esmfold.py --collect"
python scripts/08_predict_esmfold.py --collect
echo "--- 10_aggregate.py"
python scripts/10_aggregate.py
echo "--- 11_statistics.py"
python scripts/11_statistics.py
echo "--- capture_environment.py"
python scripts/capture_environment.py
echo "done"

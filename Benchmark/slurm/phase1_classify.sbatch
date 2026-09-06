#!/bin/bash
#SBATCH --job-name=bm_classify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 1, remaining curation: dssp, select, targets. Serial, one job.
# Needs a working mkdssp; the path comes from cluster.yaml paths.dssp_bin.

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

echo "host=$(hostname) job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION}"

echo "=== dssp ==="
python scripts/00_curate_scaffolds.py --stage dssp

echo "=== select ==="
python scripts/00_curate_scaffolds.py --stage select

echo "=== targets ==="
python scripts/00_curate_scaffolds.py --stage targets

echo "=== Phase 1 curation complete ==="

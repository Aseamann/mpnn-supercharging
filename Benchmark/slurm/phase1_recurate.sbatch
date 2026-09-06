#!/bin/bash
#SBATCH --job-name=bm_recurate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#
# Phase 1 rerun from filter onward, after Decisions 11 and 12 changed the chain
# filter and dropped loop_rich. pool and fetch are not repeated; all 1054
# structures are already on disk.

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

for stage in filter cluster dssp select targets; do
    echo "=== ${stage} ==="
    python scripts/00_curate_scaffolds.py --stage "${stage}"
done

echo "=== Phase 1 recuration complete ==="

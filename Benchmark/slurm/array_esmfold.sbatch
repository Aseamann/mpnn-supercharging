#!/bin/bash
#SBATCH --job-name=bm_esmfold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
#SBATCH --requeue
#
# Phase 6a. Adapted from /projects/f_sdk94_1/Tools/ESMfold2/run_esmfold2.slurm,
# per PLAN.md Section 9, which says to copy that example out and adapt it rather
# than submitting from the shared tool directory. The shared tree is read-only.
#
# Changes from the original:
#   - logs and output land under Benchmark/, not /scratch and not the tool dir
#   - --also-pdb, because per-residue pLDDT and the TM-align superposition both
#     need atom records this benchmark can parse; the tool writes mmCIF by default
#   - partition, gres, cpus, mem and time come from config/cluster.yaml via the
#     submitting driver, so nothing cluster-specific is hardcoded here
#
# fold_fasta.py shards the FASTA itself, and skips any record whose .cif already
# exists, so a resubmission of the same array resumes rather than refolding.
#
# The shard COUNT comes from BM_NUM_SHARDS, exported by the submitting driver,
# NOT from SLURM_ARRAY_TASK_COUNT. Those two agree only for a full array. On a
# partial rerun such as --array=2,28,29 the task count is 3, so deriving the
# count from it would re-shard the FASTA into 3 pieces and ask for shard 28 of 3,
# which selects no records at all: the tasks exit cleanly having folded nothing.
# That happened on 2026-08-13 and left 659 records silently unfolded. See
# logs/ISSUES.md item 13.
#
#   python scripts/08_predict_esmfold.py --emit-fasta
#   python scripts/08_predict_esmfold.py --submit --shards 40
#   python scripts/08_predict_esmfold.py --collect

set -euo pipefail

BENCHMARK_ROOT="${BM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${BENCHMARK_ROOT}" || ! -f "${BENCHMARK_ROOT}/config/cluster.yaml" ]]; then
    echo "BM_ROOT must point at the Benchmark directory (got '${BENCHMARK_ROOT}')" >&2
    exit 1
fi
cd "${BENCHMARK_ROOT}"

read_cfg () { python3 -c "import yaml,sys;d=yaml.safe_load(open('config/cluster.yaml'));print(eval(sys.argv[1],{'d':d}))" "$1"; }

TOOL_ROOT=$(read_cfg "d['paths']['esmfold']")
ESM_ENV=$(read_cfg "d['conda']['esmfold_env']")
CACHE_DIR=$(read_cfg "d['paths']['esmfold_cache']")

FOLD_SCRIPT="${TOOL_ROOT}/fold_fasta.py"
PYTHON="${ESM_ENV}/bin/python"

FASTA="${BENCHMARK_ROOT}/predictions/esmfold/designs.fasta"
OUTDIR="${BENCHMARK_ROOT}/predictions/esmfold/structures"

# Keep these in step with ESMFOLD_MODEL / ESMFOLD_SETTINGS in
# scripts/08_predict_esmfold.py, which records them into every output row.
MODEL="${MODEL:-biohub/ESMFold2-Fast}"
NUM_LOOPS="${NUM_LOOPS:-3}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-50}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-1}"
SEED="${SEED:-0}"

mkdir -p "${OUTDIR}"

if [[ ! -f "${FASTA}" ]]; then
    echo "missing ${FASTA}; run --emit-fasta first" >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "interpreter not found at ${PYTHON}" >&2
    exit 4
fi

module purge
module use /projects/community-old/modulefiles
module use /projects/community/modulefiles

export PATH="${ESM_ENV}/bin:${PATH}"
# Every cache that would otherwise land in $HOME is pinned to the project dir.
# /home/als515 is off limits, so this is not merely tidiness.
export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${CACHE_DIR}/hub"
export HF_ASSETS_CACHE="${CACHE_DIR}/assets"
export TORCH_HOME="${CACHE_DIR}/torch"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="/scratch/${USER}/tmp"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}"
umask 002

echo "==============================================================="
echo "job      : ${SLURM_JOB_ID} shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "node     : $(hostname)"
echo "fasta    : ${FASTA}"
echo "outdir   : ${OUTDIR}"
echo "model    : ${MODEL}"
echo "settings : loops=${NUM_LOOPS} steps=${NUM_SAMPLING_STEPS} samples=${NUM_DIFFUSION_SAMPLES} seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "==============================================================="

"${PYTHON}" "${FOLD_SCRIPT}" \
    --fasta "${FASTA}" \
    --outdir "${OUTDIR}" \
    --model "${MODEL}" \
    --num-loops "${NUM_LOOPS}" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
    --num-diffusion-samples "${NUM_DIFFUSION_SAMPLES}" \
    --seed "${SEED}" \
    --sort-by-length \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${BM_NUM_SHARDS}" \
    --also-pdb

echo "finished : $(date)"

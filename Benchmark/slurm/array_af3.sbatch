#!/bin/bash
#SBATCH --job-name=bm_af3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --export=NONE
#
# Phase 6b. Adapted from /projects/f_sdk94_1/Tools/AlphaFold3/af3_monomer/
# 02_af3_batch_input.sh, per PLAN.md Section 9, which says to copy the examples
# out and adapt them. The shared tool tree is read-only.
#
# Changes from the original:
#   - a Slurm array indexed into predictions/af3/subset.csv, rather than a bash
#     loop that submits one job per JSON and polls squeue to stay under a cap
#   - input and output under Benchmark/predictions/af3/, not /scratch
#   - --norun_data_pipeline, because Section 9 requires single-sequence mode.
#     The JSONs already carry empty unpairedMsa/pairedMsa and no templates; this
#     flag makes sure no MSA or template search runs anyway. Building an MSA for
#     a supercharged sequence would retrieve wild-type homologs and rescue the
#     prediction with evolutionary signal the design does not have.
#
#   python scripts/09_predict_af3_subset.py --select
#   python scripts/09_predict_af3_subset.py --submit
#   python scripts/09_predict_af3_subset.py --collect

set -euo pipefail

BENCHMARK_ROOT="${BM_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${BENCHMARK_ROOT}" || ! -f "${BENCHMARK_ROOT}/config/cluster.yaml" ]]; then
    echo "BM_ROOT must point at the Benchmark directory (got '${BENCHMARK_ROOT}')" >&2
    exit 1
fi
cd "${BENCHMARK_ROOT}"

SUBSET="predictions/af3/subset.csv"
INPUT_DIR="${BENCHMARK_ROOT}/predictions/af3/af_input"
OUTPUT_DIR="${BENCHMARK_ROOT}/predictions/af3/af_output"

if [[ ! -f "${SUBSET}" ]]; then
    echo "missing ${SUBSET}; run --select first" >&2
    exit 1
fi

# Row SLURM_ARRAY_TASK_ID of the subset, skipping the header. Column 1 is
# job_name. Quoted fields are not used in that column, so awk is safe here.
JOB=$(awk -F',' -v i="${SLURM_ARRAY_TASK_ID}" 'NR==i+2 {print $1; exit}' "${SUBSET}")
if [[ -z "${JOB}" ]]; then
    echo "no row ${SLURM_ARRAY_TASK_ID} in ${SUBSET}" >&2
    exit 1
fi
if [[ ! -f "${INPUT_DIR}/${JOB}.json" ]]; then
    echo "missing ${INPUT_DIR}/${JOB}.json" >&2
    exit 1
fi

source /etc/profile.d/modules.sh
module purge
module use /projects/community/modulefiles
module load apptainer/1.2.5
module load alphafold/vs3.0.0-pgarias

# One private input directory per job. The container binds a whole directory,
# and array tasks sharing one would race on it.
WORK="${OUTPUT_DIR}/_work/${JOB}"
mkdir -p "${WORK}/af_input" "${OUTPUT_DIR}"
cp "${INPUT_DIR}/${JOB}.json" "${WORK}/af_input/${JOB}.json"

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} name=${JOB}"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec -B "${WORK}/af_input":/root/af_input \
    -B "${OUTPUT_DIR}":/root/af_output \
    -B "${ALPHAFOLD_MODELWEIGHTS}":/root/models \
    -B "${ALPHAFOLD_DATA_PATH}":/root/public_databases \
    --pwd /app/alphafold \
    --nv "${CONTAINERDIR}/alphafold3.sif" \
    python run_alphafold.py \
    --json_path="/root/af_input/${JOB}.json" \
    --db_dir=/root/public_databases \
    --model_dir=/root/models \
    --output_dir=/root/af_output \
    --norun_data_pipeline

echo "finished: $(date)"

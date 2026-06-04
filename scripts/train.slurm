#!/bin/bash
#SBATCH --job-name=tier1_dce
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage (submit from the project root, i.e. the dir containing tier1_static/ and venv/):
#   sbatch tier1_static/scripts/train.slurm
# Override any default via environment variables, e.g.:
#   MODEL=gan EPOCHS=200 DATA_ROOT=/path/to/Bao_DCE sbatch tier1_static/scripts/train.slurm

set -euo pipefail

# ---- configurable knobs (env-overridable) -------------------------------------
MODEL=${MODEL:-ldm_flow}
DATA_ROOT=${DATA_ROOT:-/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/Bao_DCE}
OUTPUT_DIR=${OUTPUT_DIR:-runs/${MODEL}_${SLURM_JOB_ID:-local}}
EPOCHS=${EPOCHS:-100}
VAE_EPOCHS=${VAE_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
SPATIAL_SIZE=${SPATIAL_SIZE:-"32 192 192"}
MEDICALNET_WEIGHTS=${MEDICALNET_WEIGHTS:-pretrain/resnet_18.pth}
EXTRA_ARGS=${EXTRA_ARGS:-}

# ---- environment --------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# site-specific module loads (uncomment/edit as needed)
# module load cuda/12.1
# module load python/3.12

# activate the project venv
source venv/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} model=${MODEL}"
echo "data_root=${DATA_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi || true

# ---- run ----------------------------------------------------------------------
srun python -m tier1_static.main \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --vae-epochs "${VAE_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --spatial-size ${SPATIAL_SIZE} \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --medicalnet-weights "${MEDICALNET_WEIGHTS}" \
    ${EXTRA_ARGS}

echo "done -> ${OUTPUT_DIR}"

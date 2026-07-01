#!/bin/bash
#SBATCH --job-name=tier1_px2d
#SBATCH --partition=pod
#SBATCH --qos=pod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# 2D pixel-space pix2pix (the reproducibility / data-scale lever) on CHPC.
# Reuses the 3D data/loss/metrics; only the data view (2D slices) + model differ.
#   cd /mnt/scratch/user/kmadapati/tier1_dce
#   sbatch tier1_static/scripts/train2d.slurm
# Override via env vars: DATA_ROOT, OUTPUT_DIR, SPATIAL_SIZE, EPOCHS, BATCH_SIZE, BASE_CH, EXTRA_ARGS.
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-kmadapati_env}
MODEL=${MODEL:-gan}                           # gan (pix2pix) | ldm_flow (pixel-space 2D CFM)
SAMPLE_STEPS=${SAMPLE_STEPS:-50}              # flow ODE steps (ignored by gan)
DATA_ROOT=${DATA_ROOT:-/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/Bao_DCE}
OUTPUT_DIR=${OUTPUT_DIR:-runs/px2d_${SLURM_JOB_ID:-local}}
SPATIAL_SIZE=${SPATIAL_SIZE:-"32 256 256"}   # D H W; D = #slices/case, H W = native-ish in-plane
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}                  # 2D -> big batches fit
BASE_CH=${BASE_CH:-64}
LR=${LR:-2e-4}
CKPT_EVERY=${CKPT_EVERY:-5}
VAL_FRAC=${VAL_FRAC:-0.1}
DCE_NORM=${DCE_NORM:-robust}
DCE_ROBUST_K=${DCE_ROBUST_K:-3}
T2W_NORM=${T2W_NORM:-percentile}
ROI_WEIGHT=${ROI_WEIGHT:-10.0}
ADV_WEIGHT=${ADV_WEIGHT:-1.0}
PERCEPTUAL=${PERCEPTUAL:-0.1}
PERCEPTUAL_DEPTH=${PERCEPTUAL_DEPTH:-18}
MEDICALNET_WEIGHTS=${MEDICALNET_WEIGHTS:-pretrain/resnet_${PERCEPTUAL_DEPTH}.pth}
TEST_HOSPITALS=${TEST_HOSPITALS:-jiulong}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-.}"; mkdir -p logs
set +u
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"; set -u
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} px2d  data=${DATA_ROOT} out=${OUTPUT_DIR}"

srun python -m tier1_static.main2d \
  --model "${MODEL}" --sample-steps "${SAMPLE_STEPS}" \
  --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --spatial-size ${SPATIAL_SIZE} --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
  --base-ch "${BASE_CH}" --lr "${LR}" --ckpt-every "${CKPT_EVERY}" --val-frac "${VAL_FRAC}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --dce-norm "${DCE_NORM}" --dce-robust-k "${DCE_ROBUST_K}" --t2w-norm "${T2W_NORM}" \
  --roi-weight "${ROI_WEIGHT}" --adv-weight "${ADV_WEIGHT}" \
  --perceptual "${PERCEPTUAL}" --perceptual-depth "${PERCEPTUAL_DEPTH}" \
  --medicalnet-weights "${MEDICALNET_WEIGHTS}" \
  --test-hospitals ${TEST_HOSPITALS} --no-fid ${EXTRA_ARGS}
echo "done -> ${OUTPUT_DIR}"

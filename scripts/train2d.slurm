#!/bin/bash
#SBATCH --job-name=tier1_px2d
#SBATCH --partition=pod
#SBATCH --account=ucsf_pod
#SBATCH --qos=pod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# 2D holds the whole slice cache in RAM: ~52k train slices of (3+1+1,256,256) fp16
# is ~34G, plus val/test caches (~10G) and torch/CUDA overhead. 64G OOM-killed the
# first v3 attempt during slice-prep; leave headroom.
#SBATCH --mem=96G
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
# ---- UCSF cohort (set UCSF_ROOT to switch off the Bao layout entirely) ---------
# 2D is where a flow can actually be trained to convergence: slicing turns ~3.2k
# volumes into ~43k slices AND each step is ~6x cheaper, so a 24h run does ~200k
# steps instead of ~35k.
UCSF_ROOT=${UCSF_ROOT:-}                      # staged root from stage_ucsf.py
UCSF_TEST_FRAC=${UCSF_TEST_FRAC:-0.15}
FIRST_STAGE=${FIRST_STAGE:-vae}               # vae | medvae (2D: use medvae_4_1_2d)
MEDVAE_MODEL=${MEDVAE_MODEL:-medvae_4_1_2d}   # 1-CHANNEL 2D model (…_4_3_2d is 3-channel)
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-0.0}           # image-space anchor for the latent flow
ANCHOR_T_MAX=${ANCHOR_T_MAX:-0.3}
USE_PREGAD=${USE_PREGAD:-0}                   # 1 = pre-contrast T1 as a 4th cond channel
COMPUTE_FID=${COMPUTE_FID:-0}                 # 1 = FID on (needs cached Inception weights)
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-.}"; mkdir -p logs
set +u
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"; set -u
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# FIRST_STAGE=medvae loads weights from the HF cache; compute nodes have no internet,
# so point HF_HOME at the pre-downloaded cache and force offline. Without this the
# 2D MedVAE run dies with "Cannot send a request, as the client has been closed."
# (harmless for non-MedVAE runs, which never touch HF). TORCH_HOME is where
# torch_fidelity caches the Inception backbone used for FID.
export HF_HOME=${HF_HOME:-/mnt/scratch/user/kmadapati/.hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TORCH_HOME=${TORCH_HOME:-$(pwd)/.torch}
# data source: UCSF (patient-level split) or the Bao silver layout
DATA_ARGS=(--data-root "${DATA_ROOT}" --test-hospitals ${TEST_HOSPITALS})
if [ -n "${UCSF_ROOT}" ]; then
  DATA_ARGS=(--ucsf-main-root "${UCSF_ROOT}" --ucsf-test-frac "${UCSF_TEST_FRAC}")
fi
if [ "${USE_PREGAD}" = "1" ]; then DATA_ARGS+=(--use-pregad); fi
FS_ARGS=(--first-stage "${FIRST_STAGE}")
if [ "${FIRST_STAGE}" = "medvae" ]; then FS_ARGS+=(--medvae-model "${MEDVAE_MODEL}"); fi
FID_ARG=--no-fid; if [ "${COMPUTE_FID}" = "1" ]; then FID_ARG=--compute-fid; fi
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} px2d  cohort=${UCSF_ROOT:-$DATA_ROOT} out=${OUTPUT_DIR}"

srun python -m tier1_static.main2d \
  --model "${MODEL}" --sample-steps "${SAMPLE_STEPS}" \
  "${DATA_ARGS[@]}" "${FS_ARGS[@]}" --output-dir "${OUTPUT_DIR}" \
  --anchor-weight "${ANCHOR_WEIGHT}" --anchor-t-max "${ANCHOR_T_MAX}" \
  --spatial-size ${SPATIAL_SIZE} --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
  --base-ch "${BASE_CH}" --lr "${LR}" --ckpt-every "${CKPT_EVERY}" --val-frac "${VAL_FRAC}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --dce-norm "${DCE_NORM}" --dce-robust-k "${DCE_ROBUST_K}" --t2w-norm "${T2W_NORM}" \
  --roi-weight "${ROI_WEIGHT}" --adv-weight "${ADV_WEIGHT}" \
  --perceptual "${PERCEPTUAL}" --perceptual-depth "${PERCEPTUAL_DEPTH}" \
  --medicalnet-weights "${MEDICALNET_WEIGHTS}" \
  ${FID_ARG} ${EXTRA_ARGS}
echo "done -> ${OUTPUT_DIR}"

#!/bin/bash
#SBATCH --job-name=tier1_dce
#SBATCH --partition=pod
#SBATCH --qos=pod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Tier-1 synthetic DCE training on CHPC (pod partition = H200).
#
# Submit from the PROJECT ROOT (the dir holding tier1_static/ and pretrain/):
#   cd /mnt/scratch/user/kmadapati/tier1_dce
#   sbatch tier1_static/scripts/train.slurm
#
# Every knob below is env-overridable. Examples:
#   MODEL=gan sbatch tier1_static/scripts/train.slurm                 # swap model
#   MODEL=ldm_ddpm TIMESTEPS=1000 SAMPLE_STEPS=100 sbatch ...         # ddpm sampler
#   PERCEPTUAL=0.2 SSIM=2.0 ROI_WEIGHT=20 sbatch ...                  # loss sweep
#   BATCH_SIZE=4 EPOCHS=200 sbatch ...                               # H200 headroom
#   EXTRA_ARGS="--z-dim 256 --ch-mults 1 2 4 8" sbatch ...            # anything else
# Override SBATCH resources on the command line:
#   sbatch --time=04:00:00 --partition=gpu tier1_static/scripts/train.slurm

set -euo pipefail

# ---- infra --------------------------------------------------------------------
CONDA_ENV=${CONDA_ENV:-kmadapati_env}
MODEL=${MODEL:-ldm_flow}                 # ldm_flow | ldm_ddpm | gan
DATA_ROOT=${DATA_ROOT:-/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/Bao_DCE}
OUTPUT_DIR=${OUTPUT_DIR:-runs/${MODEL}_${SLURM_JOB_ID:-local}}
SEED=${SEED:-0}

# ---- data / geometry ----------------------------------------------------------
SPATIAL_SIZE=${SPATIAL_SIZE:-"32 192 192"}   # D H W
REFERENCE=${REFERENCE:-dce}                   # t2w | dce | iso
DCE_PHASE=${DCE_PHASE:-early}                 # zhongyiyuan target: early | peak | <int>
TEST_HOSPITALS=${TEST_HOSPITALS:-jiulong}     # held-out center(s), space-separated
HARMONIZE=${HARMONIZE:-1}                      # 1 = Nyul harmonization on, 0 = off
HARMONIZE_MAX=${HARMONIZE_MAX:-200}

# ---- training -----------------------------------------------------------------
EPOCHS=${EPOCHS:-100}
VAE_EPOCHS=${VAE_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-4}
CKPT_EVERY=${CKPT_EVERY:-10}

# ---- model size ---------------------------------------------------------------
BASE_CH=${BASE_CH:-32}
N_UPSAMPLES=${N_UPSAMPLES:-4}                  # GAN up/down depth
KL_WEIGHT=${KL_WEIGHT:-1e-6}                   # VAE
TIMESTEPS=${TIMESTEPS:-1000}                   # diffusion
SAMPLE_STEPS=${SAMPLE_STEPS:-50}
VAE_CKPT=${VAE_CKPT:-}                          # reuse a trained VAE (skip stage 1)

# ---- loss ---------------------------------------------------------------------
L1=${L1:-1.0}
SSIM=${SSIM:-1.0}
PERCEPTUAL=${PERCEPTUAL:-0.1}
PERCEPTUAL_DEPTH=${PERCEPTUAL_DEPTH:-18}
ADV_WEIGHT=${ADV_WEIGHT:-1.0}                  # GAN
ROI_WEIGHT=${ROI_WEIGHT:-10.0}                 # prostate-ROI emphasis (1 = off)
# weights file tracks the perceptual depth unless overridden
MEDICALNET_WEIGHTS=${MEDICALNET_WEIGHTS:-pretrain/resnet_${PERCEPTUAL_DEPTH}.pth}

# ---- eval ---------------------------------------------------------------------
COMPUTE_FID=${COMPUTE_FID:-1}                   # 1 = FID on, 0 = off
FID_SLICES=${FID_SLICES:-8}
FID_BATCH_SIZE=${FID_BATCH_SIZE:-32}

EXTRA_ARGS=${EXTRA_ARGS:-}                       # catch-all for anything not above

# ---- environment --------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Activate the conda env. Its init scripts reference unbound vars, so relax `set -u`
# just around activation. `conda` is on PATH because sbatch exports the submit env.
set +u
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} model=${MODEL} env=${CONDA_ENV}"
echo "data_root=${DATA_ROOT}  output_dir=${OUTPUT_DIR}"
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
nvidia-smi || true

# ---- assemble args ------------------------------------------------------------
ARGS=(
  --model "${MODEL}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --spatial-size ${SPATIAL_SIZE}
  --reference "${REFERENCE}"
  --dce-phase "${DCE_PHASE}"
  --test-hospitals ${TEST_HOSPITALS}
  --harmonize-max "${HARMONIZE_MAX}"
  --epochs "${EPOCHS}"
  --vae-epochs "${VAE_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --ckpt-every "${CKPT_EVERY}"
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"
  --base-ch "${BASE_CH}"
  --n-upsamples "${N_UPSAMPLES}"
  --kl-weight "${KL_WEIGHT}"
  --timesteps "${TIMESTEPS}"
  --sample-steps "${SAMPLE_STEPS}"
  --l1 "${L1}"
  --ssim "${SSIM}"
  --perceptual "${PERCEPTUAL}"
  --perceptual-depth "${PERCEPTUAL_DEPTH}"
  --medicalnet-weights "${MEDICALNET_WEIGHTS}"
  --adv-weight "${ADV_WEIGHT}"
  --roi-weight "${ROI_WEIGHT}"
  --fid-slices "${FID_SLICES}"
  --fid-batch-size "${FID_BATCH_SIZE}"
)
if [ "${HARMONIZE}" != "1" ]; then ARGS+=(--no-harmonize); fi
if [ "${COMPUTE_FID}" != "1" ]; then ARGS+=(--no-fid); fi
if [ -n "${VAE_CKPT}" ]; then ARGS+=(--vae-ckpt "${VAE_CKPT}"); fi
if [ -n "${EXTRA_ARGS}" ]; then ARGS+=(${EXTRA_ARGS}); fi

# ---- run ----------------------------------------------------------------------
srun python -m tier1_static.main "${ARGS[@]}"

echo "done -> ${OUTPUT_DIR}"

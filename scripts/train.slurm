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
T2W_NORM=${T2W_NORM:-nyul}                    # T2w INPUT norm: nyul | percentile (per-image, generalizes)
DCE_NORM=${DCE_NORM:-percentile}              # DCE target norm: percentile | robust (bladder-insensitive)
DCE_ROBUST_K=${DCE_ROBUST_K:-2.0}            # robust DCE half-width (body-spread units); tune via alignment probe

# ---- training -----------------------------------------------------------------
EPOCHS=${EPOCHS:-100}
VAE_EPOCHS=${VAE_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-4}
CKPT_EVERY=${CKPT_EVERY:-10}
VAL_FRAC=${VAL_FRAC:-0.1}          # train fraction held out for best-ckpt selection (0 = off)
VAL_EVERY=${VAL_EVERY:-0}          # epochs between val scorings (0 = use CKPT_EVERY; set 2 for GAN)
SELECT_METRIC=${SELECT_METRIC:-ssim_roi}  # best-ckpt metric: ssim_roi|roi_pearson|realism|balanced
                                          # ('balanced' for the radiologist-reading realism goal)

# ---- model size ---------------------------------------------------------------
BASE_CH=${BASE_CH:-32}
N_UPSAMPLES=${N_UPSAMPLES:-4}                  # GAN up/down depth
KL_WEIGHT=${KL_WEIGHT:-1e-6}                   # VAE
VAE_ADV_WEIGHT=${VAE_ADV_WEIGHT:-0.0}          # VAE patch-discriminator weight (0=off; 0.1-0.5 sharpens, reduces LDM blur)
VAE_ADV_WARMUP=${VAE_ADV_WARMUP:-5}            # epochs of pure recon before adversarial term starts
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-0.0}           # flow trajectory-anchoring (FlowMI-style image-space recon); 0=off, try 0.5-2.0
ANCHOR_T_MAX=${ANCHOR_T_MAX:-1.0}             # only anchor for t<this (1.0=all; 0.5=reliable low-noise regime)
COND_DIM=${COND_DIM:-0}                        # cross-attention conditioning token channels (0=off, concat-only; try 64/128)
FLOW_SOURCE=${FLOW_SOURCE:-noise}             # flow ODE start: noise | t2w (image-to-image structural prior)
VAE_JOINT=${VAE_JOINT:-0}                      # 1 = train VAE on DCE+T2w (sound encode(T2w) for FLOW_SOURCE=t2w)
FLOW_ADV_WEIGHT=${FLOW_ADV_WEIGHT:-0.0}       # flow adversarial head (cond patch-disc on decoded pred); 0=off, try 0.1
FLOW_FEAT_WEIGHT=${FLOW_FEAT_WEIGHT:-1.0}     # feature-matching weight for the flow adversarial head
FLOW_ADV_WARMUP=${FLOW_ADV_WARMUP:-10}        # epochs of pure flow before adversarial term starts
FLOW_ADV_T=${FLOW_ADV_T:-0.5}                 # noise level t for the adversarial decoded fake
TIMESTEPS=${TIMESTEPS:-1000}                   # diffusion
SAMPLE_STEPS=${SAMPLE_STEPS:-50}
X0_CLAMP=${X0_CLAMP:-3.0}                        # ddim x0 bound (latents ~[-2.4,1.4]; 0=off, garbage)
BETA_SCHEDULE=${BETA_SCHEDULE:-linear}          # ddpm noise schedule: linear | cosine
CFG_DROPOUT=${CFG_DROPOUT:-0.0}                 # classifier-free guidance: cond-dropout prob in training (0.1 to enable)
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}           # CFG sampling scale (1=off; >1 sharpens; needs CFG-trained model)
MODALITY_DROPOUT=${MODALITY_DROPOUT:-0}         # 1 = Layer-1 missing-sequence robustness (LDM cond 3->6)
MODALITY_FULL_PROB=${MODALITY_FULL_PROB:-0.3}   # prob a training sample keeps all 3 sequences
EVAL_MODALITIES=${EVAL_MODALITIES:-111}         # eval input subset bits T2w,DWI,ADC (111/100/101/...)
LATENT_CENTER=${LATENT_CENTER:-0}               # 1 = zero-center latents (recommended for ddpm)
VAE_CKPT=${VAE_CKPT:-}                          # reuse a trained VAE (skip stage 1)

# ---- loss ---------------------------------------------------------------------
L1=${L1:-1.0}
SSIM=${SSIM:-1.0}
PERCEPTUAL=${PERCEPTUAL:-0.1}
PERCEPTUAL_DEPTH=${PERCEPTUAL_DEPTH:-18}
ADV_WEIGHT=${ADV_WEIGHT:-1.0}                  # GAN
ROI_WEIGHT=${ROI_WEIGHT:-10.0}                 # prostate-ROI emphasis (1 = off)
PZ_WEIGHT=${PZ_WEIGHT:-1.0}                     # zone-aware: extra weight on peripheral zone (DCE matters here); 1=off, try 2-3
TZ_WEIGHT=${TZ_WEIGHT:-1.0}                     # zone-aware: weight on transition zone (1=off)
RADIO_WEIGHT=${RADIO_WEIGHT:-0.0}              # ClinDCE regional-radiomics loss (0=off; try 0.1)
FOCAL_WEIGHT=${FOCAL_WEIGHT:-0.0}              # ClinDCE focal-enhancement loss (0=off; try 0.1)
EMA_DECAY=${EMA_DECAY:-0.999}                  # EMA of generator/UNet weights (0=off)
# weights file tracks the perceptual depth unless overridden
MEDICALNET_WEIGHTS=${MEDICALNET_WEIGHTS:-pretrain/resnet_${PERCEPTUAL_DEPTH}.pth}

# ---- eval ---------------------------------------------------------------------
COMPUTE_FID=${COMPUTE_FID:-1}                   # 1 = FID on, 0 = off
FID_SLICES=${FID_SLICES:-8}
FID_BATCH_SIZE=${FID_BATCH_SIZE:-32}
EVAL_ONLY=${EVAL_ONLY:-0}                        # 1 = skip training, load ckpt from OUTPUT_DIR and just evaluate

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

# reduce CUDA fragmentation OOMs (3D attention allocates large transient tensors)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# torch_fidelity downloads the Inception backbone here; keep it off the (quota'd) HOME
export TORCH_HOME=${TORCH_HOME:-$(pwd)/.torch}
# --first-stage medvae loads MedVAE weights from the HF cache; compute nodes have no
# internet, so point HF_HOME at the pre-downloaded cache and force offline (harmless
# for non-MedVAE runs, which never touch HF).
export HF_HOME=${HF_HOME:-/mnt/scratch/user/kmadapati/.hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

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
  --t2w-norm "${T2W_NORM}"
  --dce-norm "${DCE_NORM}"
  --dce-robust-k "${DCE_ROBUST_K}"
  --dce-phase "${DCE_PHASE}"
  --test-hospitals ${TEST_HOSPITALS}
  --harmonize-max "${HARMONIZE_MAX}"
  --epochs "${EPOCHS}"
  --vae-epochs "${VAE_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --ckpt-every "${CKPT_EVERY}"
  --val-frac "${VAL_FRAC}"
  --val-every "${VAL_EVERY}"
  --select-metric "${SELECT_METRIC}"
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"
  --base-ch "${BASE_CH}"
  --n-upsamples "${N_UPSAMPLES}"
  --kl-weight "${KL_WEIGHT}"
  --vae-adv-weight "${VAE_ADV_WEIGHT}"
  --vae-adv-warmup "${VAE_ADV_WARMUP}"
  --anchor-weight "${ANCHOR_WEIGHT}"
  --anchor-t-max "${ANCHOR_T_MAX}"
  --cond-dim "${COND_DIM}"
  --flow-source "${FLOW_SOURCE}"
  --flow-adv-weight "${FLOW_ADV_WEIGHT}"
  --flow-feat-weight "${FLOW_FEAT_WEIGHT}"
  --flow-adv-warmup "${FLOW_ADV_WARMUP}"
  --flow-adv-t "${FLOW_ADV_T}"
  --timesteps "${TIMESTEPS}"
  --sample-steps "${SAMPLE_STEPS}"
  --x0-clamp "${X0_CLAMP}"
  --beta-schedule "${BETA_SCHEDULE}"
  --cfg-dropout "${CFG_DROPOUT}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --modality-full-prob "${MODALITY_FULL_PROB}"
  --eval-modalities "${EVAL_MODALITIES}"
  --l1 "${L1}"
  --ssim "${SSIM}"
  --perceptual "${PERCEPTUAL}"
  --perceptual-depth "${PERCEPTUAL_DEPTH}"
  --medicalnet-weights "${MEDICALNET_WEIGHTS}"
  --adv-weight "${ADV_WEIGHT}"
  --roi-weight "${ROI_WEIGHT}"
  --pz-weight "${PZ_WEIGHT}"
  --tz-weight "${TZ_WEIGHT}"
  --radio-weight "${RADIO_WEIGHT}"
  --focal-weight "${FOCAL_WEIGHT}"
  --ema-decay "${EMA_DECAY}"
  --fid-slices "${FID_SLICES}"
  --fid-batch-size "${FID_BATCH_SIZE}"
)
if [ "${HARMONIZE}" != "1" ]; then ARGS+=(--no-harmonize); fi
if [ "${COMPUTE_FID}" != "1" ]; then ARGS+=(--no-fid); fi
if [ "${EVAL_ONLY}" = "1" ]; then ARGS+=(--eval-only); fi
if [ "${LATENT_CENTER}" = "1" ]; then ARGS+=(--latent-center); fi
if [ "${MODALITY_DROPOUT}" = "1" ]; then ARGS+=(--modality-dropout); fi
if [ "${VAE_JOINT}" = "1" ]; then ARGS+=(--vae-joint); fi
if [ -n "${VAE_CKPT}" ]; then ARGS+=(--vae-ckpt "${VAE_CKPT}"); fi
if [ -n "${EXTRA_ARGS}" ]; then ARGS+=(${EXTRA_ARGS}); fi

# ---- run ----------------------------------------------------------------------
srun python -m tier1_static.main "${ARGS[@]}"

echo "done -> ${OUTPUT_DIR}"

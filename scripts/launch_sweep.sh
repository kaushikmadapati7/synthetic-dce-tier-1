#!/bin/bash
# Launch the upgraded GAN + ldm_flow + ldm_ddpm runs in one shot.
#
#   cd /mnt/scratch/user/kmadapati/tier1_dce
#   bash tier1_static/scripts/launch_sweep.sh
#
# Submit from the PROJECT ROOT (the dir holding tier1_static/ and pretrain/).
# Override the upgraded config via env, e.g. for a faster turnaround:
#   BASE_CH=48 BATCH_SIZE=2 EPOCHS=120 VAE_EPOCHS=60 bash tier1_static/scripts/launch_sweep.sh
set -euo pipefail

# ---- upgraded config (shared across all three; env-overridable) ----
export BASE_CH=${BASE_CH:-64}          # was 32 -> more capacity (VAE + UNet/generator)
export BATCH_SIZE=${BATCH_SIZE:-4}     # was 2
export SAMPLE_STEPS=${SAMPLE_STEPS:-100}  # was 50 (LDM sampler; inference-only)
export EPOCHS=${EPOCHS:-200}           # was 100
export VAE_EPOCHS=${VAE_EPOCHS:-100}   # was 50 (LDM first stage)

# ---- GPU allocation per run ----
# H200 (141 GB) for the memory-heavy two-stage LDMs at base_ch=64 / batch=4
POD=(--partition=pod --qos=pod --gres=gpu:1 --time=3-00:00:00)
# 96 GB RTX PRO 6000 for the single-stage GAN (skip the bad-mount node ggpu1-17)
GPU=(--partition=gpu --qos=gpu --gres=gpu:nvidia_rtx_pro_6000:1 --time=3-00:00:00 --exclude=ggpu1-17)

echo "config: BASE_CH=$BASE_CH BATCH_SIZE=$BATCH_SIZE SAMPLE_STEPS=$SAMPLE_STEPS EPOCHS=$EPOCHS VAE_EPOCHS=$VAE_EPOCHS"

# flow-matching LDM (primary) -> H200
MODEL=ldm_flow OUTPUT_DIR=runs/v2_ldm_flow \
  sbatch "${POD[@]}" tier1_static/scripts/train.slurm

# DDPM LDM with the linear-schedule + centering fix -> H200
MODEL=ldm_ddpm OUTPUT_DIR=runs/v2_ldm_ddpm BETA_SCHEDULE=linear LATENT_CENTER=1 \
  sbatch "${POD[@]}" tier1_static/scripts/train.slurm

# conditional GAN baseline -> RTX PRO 6000
MODEL=gan OUTPUT_DIR=runs/v2_gan \
  sbatch "${GPU[@]}" tier1_static/scripts/train.slurm

echo "submitted 3 jobs -> runs/v2_{ldm_flow,ldm_ddpm,gan}.  watch: squeue -u \$USER"

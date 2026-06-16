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
# Override any default via environment variables, e.g.:
#   MODEL=gan EPOCHS=200 sbatch tier1_static/scripts/train.slurm
#   EXTRA_ARGS="--batch-size 4 --perceptual 0" sbatch tier1_static/scripts/train.slurm
# Override SBATCH resources on the command line, e.g.:
#   sbatch --time=04:00:00 --partition=gpu tier1_static/scripts/train.slurm

set -euo pipefail

# ---- configurable knobs (env-overridable) -------------------------------------
CONDA_ENV=${CONDA_ENV:-kmadapati_env}
MODEL=${MODEL:-ldm_flow}
DATA_ROOT=${DATA_ROOT:-/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/Bao_DCE}
OUTPUT_DIR=${OUTPUT_DIR:-runs/${MODEL}_${SLURM_JOB_ID:-local}}
EPOCHS=${EPOCHS:-100}
VAE_EPOCHS=${VAE_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
SPATIAL_SIZE=${SPATIAL_SIZE:-"32 192 192"}
DCE_PHASE=${DCE_PHASE:-early}
MEDICALNET_WEIGHTS=${MEDICALNET_WEIGHTS:-pretrain/resnet_18.pth}
EXTRA_ARGS=${EXTRA_ARGS:-}

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
echo "data_root=${DATA_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
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
    --dce-phase "${DCE_PHASE}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --medicalnet-weights "${MEDICALNET_WEIGHTS}" \
    ${EXTRA_ARGS}

echo "done -> ${OUTPUT_DIR}"

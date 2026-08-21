#!/bin/bash
#SBATCH --job-name=dce_curves
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One-time pass: read every 4D DCE series and write a few KB of JSON per case
# (raw region means per phase + real timestamps). After this, choosing a target
# phase is an offline recompute over the whole cohort instead of a re-stage.
#
# Submit from the project root (the dir holding tier1_static/):
#   cd /mnt/scratch/user/kmadapati/tier1_dce
#   sbatch tier1_static/scripts/extract_curves.slurm
#
# Env knobs (every one below is actually read -- verified, unlike USE_PREGAD /
# SEED / ANCHOR_TIMES were in their scripts):
#   OUT_DIR MAX_GB LESION_THRESH TZ_LABEL PZ_LABEL MAIN_ROOT DCE_ROOT
#   CONDA_ENV OVERWRITE EXTRA_ARGS
#
# N_SHARDS is derived from the array itself, so it can never disagree with
# --array=. Override the array width on the command line and sharding follows:
#   sbatch --array=0-15 tier1_static/scripts/extract_curves.slurm

set -euo pipefail
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-kmadapati_env}"
OUT_DIR="${OUT_DIR:-curves}"
MAX_GB="${MAX_GB:-6.0}"
LESION_THRESH="${LESION_THRESH:-0.5}"
TZ_LABEL="${TZ_LABEL:-1}"
PZ_LABEL="${PZ_LABEL:-2}"
MAIN_ROOT="${MAIN_ROOT:-/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/UCSF_data/registered}"
DCE_ROOT="${DCE_ROOT:-/mnt/fac/CX000018_DS3/yanglab/Prostate_data_all/UCSF_data/registered}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
  N_SHARDS="$SLURM_ARRAY_TASK_COUNT"
else
  N_SHARDS=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
fi

OW_FLAG=""
[ "$OVERWRITE" = "1" ] && OW_FLAG="--overwrite"

echo "effective: shard=$SHARD/$N_SHARDS out_dir=$OUT_DIR max_gb=$MAX_GB"
echo "effective: lesion_thresh=$LESION_THRESH tz=$TZ_LABEL pz=$PZ_LABEL overwrite=$OVERWRITE"
echo "effective: main_root=$MAIN_ROOT"
echo "effective: dce_root=$DCE_ROOT"
echo "effective: conda_env=$CONDA_ENV extra_args=${EXTRA_ARGS:-<none>}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "commit: $(git -C tier1_static rev-parse --short HEAD 2>/dev/null || echo unknown)"

python -m tier1_static.data.extract_curves \
  --out-dir "$OUT_DIR" \
  --main-root "$MAIN_ROOT" \
  --dce-root "$DCE_ROOT" \
  --lesion-thresh "$LESION_THRESH" \
  --tz-label "$TZ_LABEL" \
  --pz-label "$PZ_LABEL" \
  --max-gb "$MAX_GB" \
  --shard "$SHARD" \
  --n-shards "$N_SHARDS" \
  $OW_FLAG $EXTRA_ARGS

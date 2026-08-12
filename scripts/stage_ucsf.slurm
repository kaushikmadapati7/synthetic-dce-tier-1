#!/bin/bash
#SBATCH --job-name=ucsf_stage
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# MEMORY: each worker holds one whole 4D series. Most are ~1 GB (35 phases) but this
# cohort has 70-phase studies at ~2.4 GB, so 8 workers need ~20 GB + copies/overhead;
# --mem=32G was OOM-killed (exit 0:125). Keep mem >= ~12 GB per cpus-per-task.
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Stage the UCSF cohort for training: extract the target DCE phase from each ~1 GB
# 4D series once and co-locate every modality per patient on scratch. Pure I/O, so
# this is a CPU array job -- each task takes a strided slice of the patient list.
#
# Submit from the PROJECT ROOT (the dir holding tier1_static/):
#   cd /mnt/scratch/user/kmadapati/tier1_dce
#   sbatch tier1_static/scripts/stage_ucsf.slurm
#
# Knobs (env-overridable):
#   OUT=/path/to/staged TARGET_TIME=120 sbatch ...
#   MIN_ENH=1.2 sbatch ...            # drop failed/mistimed injections at staging
#   sbatch --array=0-15 ...           # more shards (also bump if the cohort grows)
#
# After ALL array tasks finish, print the aggregate summary:
#   python -m tier1_static.data.stage_ucsf --report --out $OUT

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-kmadapati_env}
FAC=${FAC:-/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/UCSF_data/registered}
DCE=${DCE:-/mnt/fac/CX000018_DS3/yanglab/Prostate_data_all/UCSF_data/registered}
OUT=${OUT:-/mnt/scratch/user/${USER}/UCSF_staged}
TARGET_TIME=${TARGET_TIME:-120}     # s post pre-contrast; 120 = stable plateau
DWI_BVALUE=${DWI_BVALUE:-1000}
MIN_ENH=${MIN_ENH:-0}               # e.g. 1.2 to drop non-enhancing cases
# Multi-timepoint (Tier-2) anchors, seconds post-injection, on a COMMON time grid so
# heterogeneous protocols (26-70 phases, 1.7-13s cadence) become comparable. 0s and
# TARGET_TIME already ship as DCE_pre_to_T2W / DCE_to_T2W. Empty = single-phase only.
# Dense through wash-in (55-90s), sparse across the plateau -- that is where PI-RADS
# curve types separate. 240s is the late anchor: 92% of cases reach it, 300s only 50%.
ANCHOR_TIMES=${ANCHOR_TIMES:-}
LIMIT=${LIMIT:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

set +u
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NSHARD=${SLURM_ARRAY_TASK_COUNT:-1}
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-NA} shard=${SHARD}/${NSHARD} out=${OUT}"
echo "effective: target_time=${TARGET_TIME} min_enh=${MIN_ENH} dwi_b=${DWI_BVALUE}" \
     "anchors='${ANCHOR_TIMES:-<none, single-phase>}' extra='${EXTRA_ARGS}'"

# Preflight: /mnt/fac is not mounted on every compute node (ENODEV / "No such
# device"), which otherwise surfaces as an opaque traceback deep in a glob. Fail
# fast and name the node so the job can be resubmitted with --nodelist/--exclude.
for d in "${FAC}" "${DCE}"; do
  if [ ! -d "${d}" ]; then
    echo "FATAL: fac mount unavailable on $(hostname): ${d}" >&2
    echo "       resubmit pinned to a node that has it, e.g. --nodelist=<node>" >&2
    exit 17
  fi
done

python -m tier1_static.data.stage_ucsf \
  --main-root "${FAC}" --dce-root "${DCE}" --out "${OUT}" \
  --target-time "${TARGET_TIME}" --dwi-bvalue "${DWI_BVALUE}" --min-enh "${MIN_ENH}" \
  ${ANCHOR_TIMES:+--anchor-times ${ANCHOR_TIMES}} \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --limit "${LIMIT}" \
  --shard "${SHARD}" --num-shards "${NSHARD}" ${EXTRA_ARGS}

echo "shard ${SHARD} done -> ${OUT}"

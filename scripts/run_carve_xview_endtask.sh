#!/bin/bash
# End-task TTA sweep, scan-sharded for many small parallel jobs (<=1h each).
# Each array task = one (seed, scan-shard) and runs ALL 6 methods on that shard
# (the expensive volume load is shared across methods). Per-scan wall-clock +
# peak GPU memory are logged.
#
# Control via env + a command-line --array:
#   DATASET   = radchest | luna | ccii          (default radchest)
#   NUM_SHARDS= scan shards per seed             (default 1)
#   SMOKE     = 1 -> 6 scans, 1 shard, seed 0    (validate GPU path fast)
#
# Submit (array = 5 seeds x NUM_SHARDS - 1):
#   smoke:  DATASET=radchest SMOKE=1 sbatch --array=0-0 --time=00:30:00 scripts/run_carve_xview_endtask.sh
#   full :  DATASET=radchest NUM_SHARDS=9 sbatch --array=0-44 scripts/run_carve_xview_endtask.sh
#
#SBATCH --job-name=xview_end
#SBATCH --output=/project/6101771/ailarmz/CT-CLIP-ailar/logs/xview_end_%A_%a.out
#SBATCH --error=/project/6101771/ailarmz/CT-CLIP-ailar/logs/xview_end_%A_%a.err
#SBATCH --time=0-01:00:00
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=aip-lsigal

export PATH="/project/aip-lsigal/ailarmz/miniconda3/envs/CT-CLIP-copy/bin:$PATH"
source /project/aip-lsigal/ailarmz/miniconda3/etc/profile.d/conda.sh
conda activate CT-CLIP-copy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/6101771/ailarmz/CT-CLIP-ailar
mkdir -p logs results_carve_xview

DATASET="${DATASET:-radchest}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SMOKE="${SMOKE:-0}"
TARGET_D="${TARGET_D:-40}"               # z-depth (40=old, 240=corrected recipe)
OUTDIR="${OUTDIR:-results_carve_xview}"  # separate dir for z=240 re-baseline
IDX="${SLURM_ARRAY_TASK_ID:-0}"
MAXSCANS=""
if [ "$SMOKE" = "1" ]; then NUM_SHARDS=1; MAXSCANS="${SMOKE_MAXSCANS:---max_scans 6}"; fi
mkdir -p "$OUTDIR"

SEED=$(( IDX / NUM_SHARDS ))
SHARD=$(( IDX % NUM_SHARDS ))

CTCLIP_W="${WEIGHTS:-/datasets/ctrate/CT-CLIP-weights/models/CT-CLIP-Related/CT-CLIP_v2.pt}"
LABELS=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels
META=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata
case "$DATASET" in
  ctrate)
    # internal CT-RATE validation; load_dataset uses --labels_csv/--ctrate_meta_csv/--subset_ids.
    DATA_ROOT=/datasets/ctrate/Validation/Data/dataset/valid
    TEST_CSV=$META/valid_predicted_labels.csv          # also passed as required --test_csv
    EXTRA="--labels_csv $META/valid_predicted_labels.csv --ctrate_meta_csv $META/validation_metadata.csv --subset_ids ${SUBSET:-$PWD/paper_submission/ctrate_internal_subset360.txt}" ;;
  radchest)
    DATA_ROOT=/datasets/ctrate/Radchest
    TEST_CSV=$LABELS/test_list.csv
    EXTRA="--metadata_csv $LABELS/CT_Scan_Metadata_Complete_35747.csv ${SUBSET:+--subset_ids $SUBSET}" ;;
  luna)
    DATA_ROOT=/datasets/ctrate/LUNA/luna16/all_subsets
    TEST_CSV=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata/train_list_LUNA.csv
    EXTRA="--luna_class_name nodule" ;;
  ccii)
    DATA_ROOT="/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CC-CCII/images/demo data"
    TEST_CSV=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata/train_list_CCII.csv
    EXTRA="" ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 1 ;;
esac

METHODS_ARG=""; [ -n "${METHODS:-}" ] && METHODS_ARG="--methods $METHODS"
CARD_ARG=""; [ -n "${CARD:-}" ] && CARD_ARG="--card_mode $CARD"; [ -n "${FIXEDK:-}" ] && CARD_ARG="$CARD_ARG --fixed_k $FIXEDK"
TGT_ARG=""; [ -n "${ADAPT:-}" ] && TGT_ARG="--adapt_target $ADAPT"; [ -n "${STEPS:-}" ] && TGT_ARG="$TGT_ARG --steps $STEPS"
LR_ARG=""; [ -n "${LR:-}" ] && LR_ARG="--lr $LR"
echo "DATASET=$DATASET SEED=$SEED SHARD=$SHARD/$NUM_SHARDS SMOKE=$SMOKE TARGET_D=$TARGET_D OUTDIR=$OUTDIR METHODS=${METHODS:-all}"
python src/carve_xview_adapt.py \
  --weights "$CTCLIP_W" --seed "$SEED" --gate_tau 0.091 $METHODS_ARG \
  --dataset "$DATASET" --radchest_root "$DATA_ROOT" --test_csv "$TEST_CSV" $EXTRA \
  --target_d "$TARGET_D" --num_views "${VIEWS:-8}" \
  --num_shards "$NUM_SHARDS" --shard_idx "$SHARD" $MAXSCANS $CARD_ARG $TGT_ARG $LR_ARG \
  --out_dir "$OUTDIR"

# When all shards of a dataset finish (CPU):
#   python analysis/carve_xview_merge_shards.py --endtask_dir results_carve_xview
#   python analysis/carve_xview_endtask_agg.py --preds_dir results_carve_xview --dataset $DATASET --gate_tau 0.091

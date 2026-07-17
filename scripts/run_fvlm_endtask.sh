#!/bin/bash
#SBATCH --job-name=fvlm_end
#SBATCH --output=/project/6101771/ailarmz/CT-CLIP-ailar/logs/fvlm_end_%A_%a.out
#SBATCH --error=/project/6101771/ailarmz/CT-CLIP-ailar/logs/fvlm_end_%A_%a.err
#SBATCH --time=0-01:00:00
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=aip-lsigal
export PATH="/project/aip-lsigal/ailarmz/miniconda3/envs/CT-CLIP-copy/bin:$PATH"
source /project/aip-lsigal/ailarmz/miniconda3/etc/profile.d/conda.sh
conda activate CT-CLIP-copy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FVLM_ROOT=/project/6101771/ailarmz/fvlm
cd /project/6101771/ailarmz/CT-CLIP-ailar
mkdir -p logs
NS="${NUM_SHARDS:-1}"; IDX="${SLURM_ARRAY_TASK_ID:-0}"; SEED=$((IDX/NS)); SHARD=$((IDX%NS))
MAXS=""; [ "${SMOKE:-0}" = "1" ] && MAXS="--max_scans ${SMAXS:-4}"
META=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata
RL=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels
if [ "$DATASET" = "ctrate" ]; then
  DSARGS="--dataset ctrate --labels_csv $META/valid_predicted_labels.csv --meta_csv $META/validation_metadata.csv --subset_ids ${SUBSET:-$PWD/paper_submission/ctrate_internal_subset360.txt}"
else
  DSARGS="--dataset radchest --labels_csv $RL/test_list.csv --meta_csv $RL/CT_Scan_Metadata_Complete_35747.csv --radchest_root /datasets/ctrate/Radchest"
fi
CARD_ARG=""; [ -n "${CARD:-}" ] && CARD_ARG="--card_mode $CARD"
MARG=""; [ -n "${METHODS:-}" ] && MARG="--methods $METHODS"
EXTRA=""; [ -n "${LR:-}" ] && EXTRA="$EXTRA --lr $LR"; [ -n "${STEPS:-}" ] && EXTRA="$EXTRA --steps $STEPS"; [ -n "${ADAPT:-}" ] && EXTRA="$EXTRA --adapt_target $ADAPT"
echo "fVLM DATASET=$DATASET SEED=$SEED SHARD=$SHARD/$NS CARD=${CARD:-sum_round} METHODS=${METHODS:-all} EXTRA=$EXTRA"
python src/fvlm_endtask_adapt.py $DSARGS $MARG $CARD_ARG --seed "$SEED" --num_shards "$NS" --shard_idx "$SHARD" $MAXS $EXTRA --out_dir "$OUTDIR"

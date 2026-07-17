#!/bin/bash
#SBATCH --job-name=carve_abl
#SBATCH --output=/project/6101771/ailarmz/CT-CLIP-ailar/logs/carve_abl_%A_%a.out
#SBATCH --error=/project/6101771/ailarmz/CT-CLIP-ailar/logs/carve_abl_%A_%a.err
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=48G
#SBATCH --account=aip-lsigal
export PATH="/project/aip-lsigal/ailarmz/miniconda3/envs/CT-CLIP-copy/bin:$PATH"
source /project/aip-lsigal/ailarmz/miniconda3/etc/profile.d/conda.sh
conda activate CT-CLIP-copy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/6101771/ailarmz/CT-CLIP-ailar
META=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata
RL=/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels
W=/datasets/ctrate/CT-CLIP-weights/models/CT-CLIP-Related/CT-CLIP_v2.pt
NS=4; IDX="${SLURM_ARRAY_TASK_ID:-0}"; SEED=$((IDX/NS)); SHARD=$((IDX%NS))
if [ "$DATASET" = "ctrate" ]; then
  DS="--dataset ctrate --labels_csv $META/valid_predicted_labels.csv --ctrate_meta_csv $META/validation_metadata.csv --subset_ids $PWD/paper_submission/ctrate_internal_subset360.txt --test_csv $META/valid_predicted_labels.csv"
else
  DS="--dataset radchest --radchest_root /datasets/ctrate/Radchest --test_csv $RL/test_list.csv --metadata_csv $RL/CT_Scan_Metadata_Complete_35747.csv"
fi
python src/carve_xview_adapt.py --weights "$W" $DS --target_d 240 \
  --num_views "$VIEWS" --keep_frac "$KEEPFRAC" --methods carve_xview \
  --seed $SEED --num_shards $NS --shard_idx $SHARD --out_dir "$OUTDIR"

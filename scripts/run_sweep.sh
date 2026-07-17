#!/bin/bash
#SBATCH --job-name=carve_sweep
#SBATCH --output=/project/6101771/ailarmz/CT-CLIP-ailar/logs/sweep_%A_%a.out
#SBATCH --error=/project/6101771/ailarmz/CT-CLIP-ailar/logs/sweep_%A_%a.err
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
W=/datasets/ctrate/CT-CLIP-weights/models/CT-CLIP-Related/CT-CLIP_v2.pt
NS=2; IDX="${SLURM_ARRAY_TASK_ID:-0}"; SEED=$((IDX/NS)); SHARD=$((IDX%NS))
python src/carve_xview_adapt.py --weights "$W" --dataset ctrate \
  --labels_csv $META/valid_predicted_labels.csv --ctrate_meta_csv $META/validation_metadata.csv \
  --subset_ids $PWD/paper_submission/ctrate_internal_subset360.txt --test_csv $META/valid_predicted_labels.csv \
  --target_d 240 --methods carve_xview \
  --num_views ${VIEWS:-8} --keep_frac ${RHO:-0.25} --lambda_neg ${LAM:-0.8} \
  --aug_hu_std ${HU:-0.02} --aug_crop ${CROP:-0.03} \
  --seed $SEED --num_shards $NS --shard_idx $SHARD --out_dir "$OUTDIR"

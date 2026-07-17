#!/bin/bash
# Launch the full Table-1 matrix at z=40 AND z=240 with REAL (varying) seeds.
# CT-CLIP: RAD-ChestCT(zeroshot) + CC-CCII(zs/vf/cf) + LUNA(zs/vf/cf), 6 methods, 5 seeds.
# Run from the repo root on the login node:  bash scripts/submit_matrix_z40_z240.sh
set -u
WD=/datasets/ctrate/CT-CLIP-weights/models/CT-CLIP-Related
MM6="zeroshot,tent,ml_tta,bem,carve_xview,carve_xview_gate"
declare -A WT=( [zeroshot]="$WD/CT-CLIP_v2.pt" [vocabfine]="$WD/CT_VocabFine_v2.pt" [classfine]="$WD/CT_LiPro_v2.pt" )

submit () {  # dataset variant depth num_shards
  local ds=$1 var=$2 d=$3 ns=$4
  local last=$(( 5*ns - 1 ))
  local out=results_matrix_${ds}_${var}_z${d}
  echo "submit $ds $var z=$d  shards=$ns  array=0-$last  -> $out"
  TARGET_D=$d DATASET=$ds NUM_SHARDS=$ns WEIGHTS="${WT[$var]}" METHODS="$MM6" OUTDIR=$out \
    sbatch --parsable --array=0-$last scripts/run_carve_xview_endtask.sh
}

for D in 40 240; do
  submit radchest zeroshot  $D 4
  for V in zeroshot vocabfine classfine; do
    submit ccii $V $D 2
    submit luna $V $D 5
  done
done
echo "all matrix cells submitted"
squeue -u ailarmz -h -r 2>/dev/null | wc -l

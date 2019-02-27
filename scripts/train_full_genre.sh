#!/usr/bin/env bash

#outdir=./output/genre
#inpaint_path=/path/to/trained/inpaint.pt
outdir=/media/Data/dsl-course/GenRe_Training/output/genre
inpaint_path=/media/Data/dsl-course/GenRe_Training/output/inpaint/depth_pred_with_sph_inpaint_ycb_genre_0.0001_all/0/best.pt
if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

source activate shaperecon

python train.py \
    --net genre_full_model \
    --pred_depth_minmax \
    --dataset ycb_genre \
    --classes "$class" \
    --batch_size 16 \
    --epoch_batches 350 \
    --eval_batches 10 \
    --log_time \
    --optim adam \
    --lr 1e-4 \
    --epoch 1000 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}' \
    --tensorboard \
    --surface_weight 10 \
    --inpaint_path "$inpaint_path" \
    $*

source deactivate

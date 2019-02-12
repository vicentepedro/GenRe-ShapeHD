#!/usr/bin/env bash

outdir=/media/Data/dsl-course/GenRe_Training/output/inpaint
net1_path=/media/Data/dsl-course/GenRe_Training/output/marrnet1/marrnet1_ycb_genre_0.001_all/0/best.pt
#net1_path=/path/to/trained/marrnet1.pt

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
    --net depth_pred_with_sph_inpaint \
    --pred_depth_minmax \
    --dataset ycb_genre \
    --classes "$class" \
    --batch_size 4 \
    --epoch_batches 2000 \
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
    --net1_path "$net1_path" \
    $*

source deactivate

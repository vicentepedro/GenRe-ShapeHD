#!/usr/bin/env bash

# Test ShapeHD

out_dir="/media/Data/dsl-course/GenRe_Testing/output/shapehd_paper/"
net1=./downloads/models/marrnet1_with_minmax.pt # Pre-trained model
net2=./downloads/models/shapehd.pt              # Pre-trained model

rgb_pattern='/media/Data/dsl-course/affordances_dataset/all_objects_hook_draw/'"$2"'/rgb/*.jpg'
mask_pattern='/media/Data/dsl-course/affordances_dataset/all_objects_hook_draw/'"$2"'/mask/*.tif'

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu obj_name [ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments
shift # obj_name / folder to test
set -e


source activate shaperecon

python 'test.py' \
    --net shapehd \
    --net_file "$net2" \
    --marrnet1_file "$net1" \
    --input_rgb "$rgb_pattern" \
    --input_mask "$mask_pattern" \
    --output_dir "$out_dir" \
    --suffix '{net}' \
    --overwrite \
    --workers 1 \
    --batch_size 1 \
    --vis_workers 4 \
    --gpu "$gpu" \
    $*

source deactivate


#!/usr/bin/env bash

# Test GenRe


out_dir="/media/Data/dsl-course/GenRe_Testing/output/genre_paper/"
fullmodel=./downloads/models/full_model.pt # Pre-trained Model

rgb_pattern='/media/Data/dsl-course/affordances_dataset/all_objects_hook_draw/'"$2"'/rgb/*.jpg'
mask_pattern='/media/Data/dsl-course/affordances_dataset/all_objects_hook_draw/'"$2"'/mask/*.tif'

echo "$out_dir"
echo "$rgb_pattern"
echo "$mask_pattern"
if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu obj_name[ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments
shift # obj_name / folder to test
set -e

source activate shaperecon

python 'test.py' \
    --net genre_full_model \
    --net_file "$fullmodel" \
    --input_rgb "$rgb_pattern" \
    --input_mask "$mask_pattern" \
    --output_dir "$out_dir" \
    --suffix '{net}' \
    --overwrite \
    --workers 0 \
    --batch_size 1 \
    --vis_workers 4 \
    --gpu "$gpu" \
    $*

source deactivate

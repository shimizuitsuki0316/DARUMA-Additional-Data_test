#!/bin/bash

input_dir="./input_data/test_data1_single"
output_dir="./out_put/daruma_single"

mkdir -p "$output_dir"

for file in "$input_dir"/*; do
    echo "Processing $file"
    filename=$(basename "$file")
    ac="${filename%.*}"
    python3 ./../daruma/cpu/predict.py "$file" -o "./out_put/daruma_single/${ac}.out"
done

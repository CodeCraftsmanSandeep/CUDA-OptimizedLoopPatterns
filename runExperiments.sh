#!/bin/bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: ./runExperiments.sh <exec_dir> <input_dir> <output_dir>"
    exit 1
fi

EXEC_DIR="$1"
INPUT_DIR="$2"
RESULTS_DIR="$3"

mkdir -p "$RESULTS_DIR"

GRID_RES="$RESULTS_DIR/gridStrideResults.csv"
BLOCK_RES="$RESULTS_DIR/blockStrideResults.csv"
WARP_RES="$RESULTS_DIR/warpStrideResults.csv"
SPEEDUP_RES="$RESULTS_DIR/speedUpResults.csv"

# Write headers with Result instead of Sum
echo "File-name,N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$GRID_RES"
echo "File-name,N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$BLOCK_RES"
echo "File-name,N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$WARP_RES"
echo "File-name,Speedup_BlockOverGrid,Speedup_WarpOverGrid" > "$SPEEDUP_RES"

for input_file in "$INPUT_DIR"/*.txt; do
    echo "Processing $input_file"

    # Run and append results
    "$EXEC_DIR/gridStride.out" "$input_file" | tail -n 1 >> "$GRID_RES"
    "$EXEC_DIR/blockStride.out" "$input_file" | tail -n 1 >> "$BLOCK_RES"
    "$EXEC_DIR/warpStride.out" "$input_file" | tail -n 1 >> "$WARP_RES"
done

# Sort the results files based on the N column (2nd column, numeric)
for file in "$GRID_RES" "$BLOCK_RES" "$WARP_RES"; do
    header=$(head -n1 "$file")
    (echo "$header" && tail -n +2 "$file" | sort -t',' -k2,2n) > "$file.tmp" && mv "$file.tmp" "$file"
done

# Extract speedup factors (grid_mean / block_mean and grid_mean / warp_mean)
paste -d',' <(tail -n +2 "$GRID_RES") \
             <(tail -n +2 "$BLOCK_RES") \
             <(tail -n +2 "$WARP_RES") | \
while IFS=',' read -r \
    grid_file grid_n grid_result grid_runs grid_mean grid_med grid_std \
    block_file block_n block_result block_runs block_mean block_med block_std \
    warp_file warp_n warp_result warp_runs warp_mean warp_med warp_std
do
    block_speedup=$(awk -v g="$grid_mean" -v b="$block_mean" 'BEGIN {printf "%.4f", g/b}')
    warp_speedup=$(awk -v g="$grid_mean" -v w="$warp_mean" 'BEGIN {printf "%.4f", g/w}')
    echo "$grid_file,$block_speedup,$warp_speedup" >> "$SPEEDUP_RES"
done

echo "✅ All results written to $RESULTS_DIR/"


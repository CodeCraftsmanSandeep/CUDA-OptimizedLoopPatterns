#!/bin/bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: ./runStateOfTheArt.sh <exec_dir> <input_dir> <output_dir>"
    exit 1
fi

EXEC_DIR="$1"
INPUT_DIR="$2"
RESULTS_DIR="$3"

mkdir -p "$RESULTS_DIR"

THRUST_RES="$RESULTS_DIR/ThrustResults.csv"
CUB_RES="$RESULTS_DIR/CUBResults.csv"

# Write headers
echo "File-name,N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$THRUST_RES"
echo "File-name,N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$CUB_RES"

for input_file in "$INPUT_DIR"/*.txt; do
    echo "Processing $input_file"

    # Run Thrust reduction and append last line to results
    "$EXEC_DIR/thrustReduction/thrustReduction.out" "$input_file" | tail -n 1 >> "$THRUST_RES"

    # Run CUB reduction and append last line to results
    "$EXEC_DIR/CUBReduction/CUBReduction.out" "$input_file" | tail -n 1 >> "$CUB_RES"
done

# Sort result files based on N column (2nd column numeric)
for file in "$THRUST_RES" "$CUB_RES"; do
    header=$(head -n1 "$file")
    (echo "$header" && tail -n +2 "$file" | sort -t',' -k2,2n) > "$file.tmp" && mv "$file.tmp" "$file"
done

echo "✅ Thrust and CUB results written to $RESULTS_DIR/"

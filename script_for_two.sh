#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <executable> <result-file>"
  exit 1
fi

EXECUTABLE="$1"
OUTPUT_FILE="$2"
INPUT_DIR="inputs"

# Temporary file to store unsorted results
TEMP_RESULTS=$(mktemp)

# Loop over all int1_*.txt
for f1 in "$INPUT_DIR"/int1_*.txt; do
  # get the numeric part after int1_
  base="${f1##*/int1_}"       # e.g. "125.txt"
  f2="$INPUT_DIR/int2_${base}"  # e.g. "inputs/int2_125.txt"

  # skip if counterpart missing
  if [[ ! -f "$f2" ]]; then
    echo "Warning: missing $f2 for $f1, skipping"
    continue
  fi

  echo "Processing pair: $(basename "$f1") & $(basename "$f2")"

  # run your executable with both files
  result_line=$("$EXECUTABLE" "$f1" "$f2" | sed -n '2p')

  # append to temp file
  echo "$result_line" >> "$TEMP_RESULTS"
done

# Sort the temp results based on column 2 (N), numerically
# Assumes CSV format: File-name,N,...
sorted_results=$(sort -t, -k2,2n "$TEMP_RESULTS")

# Write header and sorted data to output file
printf "N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)\n" > "$OUTPUT_FILE"
echo "$sorted_results" >> "$OUTPUT_FILE"

# Clean up
rm "$TEMP_RESULTS"

echo "✅ Sorted results saved to $OUTPUT_FILE"


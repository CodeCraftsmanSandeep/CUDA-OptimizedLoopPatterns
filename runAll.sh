#!/usr/bin/env bash
set -euo pipefail

# Usage: bash runAll.sh [--Module1] [--Module2] ... [--Standard] --input_dir=INPUT_DIR --output_dir=OUTPUT_DIR
# Example: bash runAll.sh --DotProduct --Standard --input_dir=../inputs --output_dir=../out

# Default "standard" sizes (multiplied by 1e7):
standard_ns=(2000000 10000000 100000000 400000000 600000000 800000000 1000000000 1250000000)

# Parse arguments
modules=()
use_standard=false
input_dir=""
output_dir=""
for arg in "$@"; do
  case "$arg" in
    --input_dir=*) input_dir="${arg#*=}" ;;  
    --output_dir=*) output_dir="${arg#*=}" ;;  
    --Standard) use_standard=true ;;
    --*) 
      name="${arg#--}"
      # Skip known args
      if [[ "$name" != input_dir && "$name" != output_dir && "$name" != Standard ]]; then
        modules+=("$name")
      fi
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# Validate input/output dirs
if [[ -z "$input_dir" || -z "$output_dir" ]]; then
  echo "Error: --input_dir and --output_dir must be specified" >&2
  exit 1
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input_dir '$input_dir' does not exist" >&2
  exit 1
fi

# Prepare output directory
mkdir -p "$output_dir"

# Build each module
for module in "${modules[@]}"; do
  echo "Building module: $module"
  make "$module"
done

# Helper: get list of input files based on --Standard flag
get_input_files() {
  if [[ "$use_standard" == true ]]; then
    for n in "${standard_ns[@]}"; do
      file="$input_dir/int_${n}.txt"
      if [[ -f "$file" ]]; then
        echo "$file"
      else
        echo "Warning: Standard input file not found: $file" >&2
      fi
    done
  else
    for file in "$input_dir"/int_*.txt; do
      [[ -f "$file" ]] && echo "$file"
    done
  fi
}

# Process each module
for module in "${modules[@]}"; do
  echo "Processing module: $module"
  # Find executables (.out)
  mapfile -t exes < <(find "$module" -type f -name '*.out')
  if [[ ${#exes[@]} -eq 0 ]]; then
    echo "Warning: No executables found in module '$module'" >&2
    continue
  fi

  # For each executable, collect results
  for exe in "${exes[@]}"; do
    exe_name=$(basename "$exe" .out)
    res_file="$output_dir/${exe_name}Results.csv"
    echo "Creating result file: $res_file"
    # Header for each results file
    echo "n,dotProduct,mean-time(ms),median-time(ms),std-deviation(ms)" > "$res_file"

    # Run for each input
    while read -r input_file; do
      echo "  Running: $exe $input_file $input_file"
      # Run executable (redirect stderr to /dev/null)
      output=$("$exe" "$input_file" "$input_file" 2>/dev/null)
      # Extract second line of output
      line=$(echo "$output" | sed -n '2p')
      echo "$line" >> "$res_file"
    done < <(get_input_files)
  done
done

# Generate comparisons.csv
comp_file="$output_dir/comparisons.csv"
echo "Creating comparison file: $comp_file"

# Build header
header="n"
for module in "${modules[@]}"; do
  mapfile -t exes < <(find "$module" -type f -name '*.out')
  for exe in "${exes[@]}"; do
    exe_name=$(basename "$exe" .out)
    header+",${exe_name}-mean(ms)"
  done
done

echo "$header" > "$comp_file"

# Determine unique sorted n values from first results file
first_module="${modules[0]}"
first_exe=$(find "$first_module" -type f -name '*.out' | head -n1)
first_res="$output_dir/$(basename "$first_exe" .out)Results.csv"
if [[ -f "$first_res" ]]; then
  mapfile -t ns < <(tail -n +2 "$first_res" | cut -d',' -f1 | sort -n | uniq)
else
  echo "Error: Cannot find first results file: $first_res" >&2
  exit 1
fi

# Populate comparisons
for n in "${ns[@]}"; do
  line="$n"
  for module in "${modules[@]}"; do
    mapfile -t exes < <(find "$module" -type f -name '*.out')
    for exe in "${exes[@]}"; do
      exe_name=$(basename "$exe" .out)
      res_file="$output_dir/${exe_name}Results.csv"
      if [[ -f "$res_file" ]]; then
        val=$(grep "^${n}," "$res_file" | cut -d',' -f3)
      else
        val=""
      fi
      line+=",$val"
    done
  done
  echo "$line" >> "$comp_file"
done

echo "All done. Results saved in $output_dir"


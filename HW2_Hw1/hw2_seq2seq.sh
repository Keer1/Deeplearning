#!/usr/bin/env bash
# HW2: Seq2Seq testing wrapper
# Usage: ./hw2_seq2seq.sh <data_dir> <output_file>
# Example:
#   ./hw2_seq2seq.sh data/testing_data testset_output.txt
#   ./hw2_seq2seq.sh data/ta_review_data tareviewset_output.txt

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <data_dir> <output_file>"
  exit 1
fi

DATA_DIR="$1"
OUTFILE="$2"

python3 test_seq2seq.py "$DATA_DIR" "$OUTFILE"


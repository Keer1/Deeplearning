

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <data_dir> <output_file>"
  exit 1
fi

DATA_DIR="$1"
OUTFILE="$2"

python3 test_seq2seq.py "$DATA_DIR" "$OUTFILE"


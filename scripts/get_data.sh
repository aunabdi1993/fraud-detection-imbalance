#!/usr/bin/env bash
# Fetch the ULB European credit-card fraud dataset.
# The CSV is ~150 MB, over GitHub's 100 MB file limit, so it is NOT committed.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

if [ -f "$DEST/creditcard.csv" ]; then
  echo "Dataset already present at $DEST/creditcard.csv"
  exit 0
fi

echo "Downloading via Kaggle API (requires ~/.kaggle/kaggle.json)..."
if command -v kaggle >/dev/null 2>&1; then
  kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DEST" --unzip
else
  cat <<'MSG'
Kaggle CLI not found. Either:
  pip install kaggle && kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
Or download manually from
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and unzip creditcard.csv into data/raw/
MSG
  exit 1
fi

echo "Verifying..."
python - <<'PY'
import hashlib, pathlib
p = pathlib.Path("data/raw/creditcard.csv")
h = hashlib.sha256()
with p.open("rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
expected = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
print("sha256:", h.hexdigest())
print("MATCHES reference copy" if h.hexdigest() == expected else "WARNING: differs from reference copy")
PY

#!/usr/bin/env bash
# Fetch the four classic MNIST IDX files (LeCun, Cortes & Burges) and gunzip
# them into ${MNIST_DIR:-$HOME/.cache/tei-fabric/mnist}/.
#
# Primary mirror: CVDF (storage.googleapis.com/cvdf-datasets/mnist/)
# Fallback:       ossci-datasets.s3.amazonaws.com/mnist/
#
# Consumed by crates/sim/tei-sim-crossbar (src/idx.rs Dataset loader) for the
# MNIST accuracy-vs-noise demo (docs/SIM-ROADMAP.md §3.3 stretch goal).
set -euo pipefail

DIR="${MNIST_DIR:-$HOME/.cache/tei-fabric/mnist}"
PRIMARY="https://storage.googleapis.com/cvdf-datasets/mnist"
FALLBACK="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

mkdir -p "$DIR"

for f in "${FILES[@]}"; do
  if [[ -s "$DIR/$f" ]]; then
    echo "ok      $DIR/$f (already present)"
    continue
  fi
  gz="$DIR/$f.gz"
  if ! curl -fsSL --retry 3 -o "$gz" "$PRIMARY/$f.gz"; then
    echo "primary mirror failed for $f.gz — trying fallback" >&2
    curl -fsSL --retry 3 -o "$gz" "$FALLBACK/$f.gz"
  fi
  gunzip -f "$gz"
  echo "fetched $DIR/$f"
done

echo "MNIST ready in $DIR"

#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
  aarch64) INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
  *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

URL="https://repo.anaconda.com/miniconda/$INSTALLER"
echo "Installing Miniconda for $ARCH at $CONDA_DIR"

rm -rf "$CONDA_DIR"
curl -fsSLo "$INSTALLER" "$URL"
bash "$INSTALLER" -b -p "$CONDA_DIR"
rm -f "$INSTALLER"

# initialize for bash
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda init bash

echo "Done. Run: source ~/.bashrc"

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
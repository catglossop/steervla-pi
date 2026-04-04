#!/bin/bash
set -euo pipefail

cd /home/noam/steervla-pi
export PATH="$HOME/.local/bin:$PATH"

# Install crcmod for fast GCS transfers if not already built with C extension.
if ! python3 -c "import crcmod; assert crcmod._usingExtension" &>/dev/null; then
    echo "Installing native crcmod..."
    sudo apt-get install -y gcc python3-dev python3-setuptools
    sudo pip3 uninstall -y crcmod || true
    sudo pip3 install --no-cache-dir -U crcmod
else
    echo "Native crcmod already installed, skipping."
fi

# Install uv only if not already present.
if ! command -v uv &>/dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "UV already installed, skipping."
fi

# Only run uv sync if the venv doesn't exist or pyproject.toml is newer than the venv.
if [ ! -d .venv ] || [ pyproject.toml -nt .venv ]; then
    echo "Syncing virtual environment..."
    GIT_LFS_SKIP_SMUDGE=1 uv sync
else
    echo "Virtual environment up to date, skipping sync."
fi

WANDB_API_KEY="${1:?WANDB_API_KEY is required as first argument}"
HF_TOKEN="${2:?HF_TOKEN is required as second argument}"

source .venv/bin/activate

# Log in only if not already authenticated.
echo "Logging into Weights & Biases..."
uv run wandb login "$WANDB_API_KEY"

echo "Logging into Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

echo "Starting training..."
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_compilation_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_LOG_COMPILES=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_steervla_simlingo \
    --exp-name=steervla_pi05_experiment --overwrite

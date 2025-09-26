#!/usr/bin/env bash
set -e
# run.sh - install (optionally) and run the full pipeline using CUDA:0

# Make sure this project folder is the current working directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project dir: $PROJECT_DIR"

# Use CUDA:0
export CUDA_VISIBLE_DEVICES=0
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# OPTIONAL: install required packages (uncomment if running first time)
# Note: this will install heavy ML libs. Make sure you pick a torch build compatible with your CUDA.
# pip install -r requirements.txt

# Quick note for colab: you may need special install steps for vllm/triton/unsloth as in your original snippet.
# If you are running on Colab/T4, follow the install instructions in your original snippet before running this.

# Run the pipeline end-to-end with moderate defaults
# This runs run.py all (which calls download->prep->baseline->sft->grpo->eval)
python run.py all


echo "All done. Check checkpoints/ and experiments/ for outputs."

## RL_GRPO_MEDMCQA

This project rebuilds the MedMCQA reinforcement learning recipe described in the GRPO paper. We start from the openlifescience MedMCQA release, clean it, convert it to a reasoning-friendly chat format, train LoRA adapters with supervised fine-tuning, and then run GRPO with reward shaping to align the model’s reasoning traces and final answers. Evaluation reuses the same formatting logic to score the baseline model, the SFT LoRA, and the GRPO LoRA on the validation split.

---

## Environment

- Python 3.10+
- CUDA-ready GPU (24 GB VRAM recommended for the GRPO stage)
- Packages listed in `requirements.sh` (installs transformers, TRL, Unsloth, vLLM, etc.)

Set up a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash requirements.sh
```

Adjust the torch/transformers wheels if your CUDA version differs from the default.

---

## How the pipeline runs

`run.py` wraps every stage. The `all` target executes them in sequence; individual stages can be called for debugging.

```bash
python run.py download   # fetch raw MedMCQA and write processed JSONL files
python run.py prep       # produce chat-style prompts with reasoning markers
python run.py baseline   # optional: score the base model on labeled validation data
python run.py sft        # LoRA SFT on the train split
python run.py grpo       # GRPO training with reward functions from rewards.py
python run.py eval       # quick validation sweep using the saved adapters

python run.py all        # convenience alias that triggers everything above
```

What each phase accomplishes:

- **Download / Prep** – Normalizes question/options, resolves gold labels, and bakes reasoning/solution tags into prompts so that every downstream step can parse answers consistently.
- **Baseline** – Runs the base model (optionally through Unsloth) to track accuracy before any training.
- **SFT** – Trains a LoRA adapter with causal language modeling on the formatted train split, teaching the model to output reasoning blocks and the final option tag.
- **GRPO** – Loads the SFT adapter, samples generations with vLLM, scores them with the handcrafted rewards, and updates the adapter with TRL’s GRPOTrainer.
- **Eval** – Loads the saved adapters, generates validation answers, extracts the `<SOLUTION>` label, and writes JSONL logs plus accuracy summaries.

Artifacts land in predictable folders: processed and formatted data under `data/`, adapters in `checkpoints/`, and evaluation traces in `experiments*/`.

---

## Evaluation details

`eval.py` forces single-GPU execution (`CUDA_VISIBLE_DEVICES=0`) and can load either or both of the LoRA adapters. It saves raw generations and the extracted final answer for each validation example.

```bash
python eval.py \
  --formatted_jsonl data/medmcqa_formatted.validation.jsonl \
  --model_name unsloth/Qwen3-4B-Base \
  --preformat_lora checkpoints/preformat_lora \
  --grpo_lora checkpoints/grpo_lora \
  --out_dir experiments_v2 \
  --batch_size 8
```

The script reports accuracy for every head you evaluate (baseline, SFT, GRPO) and writes the per-example results to `out_dir/eval_results.jsonl`.

---

## Reward shaping

`rewards.py` extracts the `<SOLUTION>` tag, checks it against the gold label, and optionally measures similarity between the generated reasoning block and any explanation text shipped with the dataset. If the sentence-transformer model is unavailable, the code falls back to token-overlap scores, so GRPO can still run.

---

## Adapting the recipe

- Trim the dataset or pass `--limit` to the baseline step for smoke tests.
- Tweak `CONFIG` inside `run.py` to change LoRA rank, max GRPO steps, learning rates, or batch sizes.
- Add new reward components to `rewards.py` and plug them into `check_answer` for custom reward mixes.
- Swap in a different base model by overriding `CONFIG["model_name"]`; adjust memory-related knobs accordingly.

---

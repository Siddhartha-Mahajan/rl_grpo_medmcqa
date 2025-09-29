python eval.py \
  --formatted_jsonl data/medmcqa_formatted.validation.jsonl \
  --model_name unsloth/Qwen3-4B-Base \
  --preformat_lora checkpoints/preformat_lora \
  --grpo_lora checkpoints/grpo_lora

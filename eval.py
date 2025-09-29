#!/usr/bin/env python3
"""
eval.py (single-GPU enforced: cuda:0)

Evaluate MedMCQA formatted dataset using:
  - Baseline (base model, no LoRA)
  - SFT LoRA (preformat LoRA)
  - GRPO LoRA (grpo-trained LoRA)

This version forces usage of CUDA device 0 and **will evaluate the ENTIRE validation file passed** (ignores any --limit).
It also saves full model outputs (including reasoning traces) per example.

Usage example:
  python eval.py --formatted_jsonl data/medmcqa_formatted.validation.jsonl \
                 --model_name unsloth/Qwen3-4B-Base \
                 --preformat_lora checkpoints/preformat_lora \
                 --grpo_lora checkpoints/grpo_lora \
                 --out_dir experiments \
                 --batch_size 8
"""
import os
# ---- FORCE single GPU visibility BEFORE any torch/transformers import ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # make only GPU 0 visible to this process

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
# set torch device to cuda:0 explicitly if available
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Try unsloth
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception:
    FastLanguageModel = None
    HAS_UNSLOTH = False

from transformers import AutoTokenizer

# Try optional imports
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PeftModel = None
    PEFT_AVAILABLE = False

# --------------------------
# Utilities
# --------------------------
def makedirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def extract_solution_from_text(text: str, sol_start: str, sol_end: str):
    if text is None:
        return None
    s = text.find(sol_start)
    if s == -1:
        return None
    s2 = text.find(sol_end, s + len(sol_start))
    if s2 == -1:
        return text[s + len(sol_start):].strip()
    return text[s + len(sol_start):s2].strip()

def build_chat_template(tokenizer):
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '" + system_prompt + "' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}" + reasoning_start + "{% endif %}"
    )

    try:
        tokenizer.chat_template = chat_template
    except Exception:
        setattr(tokenizer, "chat_template", chat_template)
    return {
        "system_prompt": system_prompt,
        "reasoning_start": reasoning_start,
        "reasoning_end": reasoning_end,
        "solution_start": solution_start,
        "solution_end": solution_end,
    }

def apply_chat_template_to_messages(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize)
        except Exception:
            pass

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        system_prompt = (
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between <start_working_out> and <end_working_out>.\n"
            "Then, provide your solution between <SOLUTION></SOLUTION>"
        )
    else:
        try:
            if "{{ '" in chat_template and "' + eos_token" in chat_template:
                system_prompt = chat_template.split("{{ '")[1].split("' +")[0]
            else:
                system_prompt = None
        except Exception:
            system_prompt = None
        if system_prompt is None:
            system_prompt = (
                "You are given a problem.\n"
                "Think about the problem and provide your working out.\n"
                "Place it between <start_working_out> and <end_working_out>.\n"
                "Then, provide your solution between <SOLUTION></SOLUTION>"
            )

    pieces = []
    if messages and messages[0].get("role") == "system":
        pieces.append(messages[0]["content"])
        loop_messages = messages[1:]
    else:
        pieces.append(system_prompt)
        loop_messages = messages

    for m in loop_messages:
        pieces.append(m.get("content", ""))

    if add_generation_prompt:
        pieces.append("<start_working_out>")

    raw = "\n".join(pieces)

    if tokenize:
        try:
            enc = tokenizer(raw, return_tensors="pt")
            if "input_ids" in enc:
                return enc["input_ids"][0].tolist()
            return enc
        except Exception:
            return raw

    return raw

# --------------------------
# Loading formatted rows (ENTIRE validation enforced)
# --------------------------
def load_formatted(formatted_jsonl: str) -> List[Dict[str, Any]]:
    rows = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows

# --------------------------
# Generate with transformers (BATCHED) -> returns full generated suffix per example (including reasoning traces)
# --------------------------
def generate_with_transformers(model_name: str, tokenizer_name: str, prompts: List[str], lora_path: Optional[str] = None,
                               max_new_tokens: int = 128, device: str = torch_device, batch_size: int = 8) -> List[str]:
    """
    Batched generation using transformers + optional PEFT LoRA.
    - Loads model onto single device (no device_map="auto").
    - Generates in batches to control memory.
    - Returns list of generated suffix strings (the model outputs only).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    # load base model and move to device
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print("Warning: normal model load failed, retrying with low_cpu_mem_usage fallback:", e)
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        model.to(device)

    # attach LoRA if requested
    if lora_path and os.path.isdir(lora_path):
        if not PEFT_AVAILABLE:
            print("PEFT not available; cannot load LoRA adapter:", lora_path)
        else:
            try:
                model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
                model.to(device)
                print("Loaded LoRA adapter:", lora_path)
            except Exception as e:
                print("Failed to load LoRA adapter:", e)

    model.eval()
    outputs: List[str] = []

    # tokenize/generate in batches
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
            )

        # decode per-example and strip prompt-decoded prefix to keep only generated suffix
        for i in range(len(batch_prompts)):
            decoded = tokenizer.decode(gen_ids[i], skip_special_tokens=False)
            prompt_decoded = tokenizer.decode(enc["input_ids"][i], skip_special_tokens=False)
            if decoded.startswith(prompt_decoded):
                out = decoded[len(prompt_decoded):].strip()
            else:
                out = decoded.strip()
            outputs.append(out)

    # if model/device assignment used shared resources, free cache
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return outputs

# --------------------------
# Generate with unsloth (keeps model's returned full outputs)
# --------------------------
def generate_with_unsloth(model_name: str, prompts: List[str], lora_path: Optional[str] = None, max_new_tokens: int = 128) -> List[str]:
    if not HAS_UNSLOTH:
        raise RuntimeError("unsloth not available")
    from vllm import SamplingParams
    # load unsloth model (unsloth uses CUDA_VISIBLE_DEVICES)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.9,
    )
    if lora_path and os.path.isdir(lora_path):
        try:
            model.load_lora(lora_path)
            print("Loaded unsloth LoRA from", lora_path)
        except Exception as e:
            print("unsloth load_lora failed:", e)

    sampling_params = SamplingParams(temperature=1.0, max_tokens=max_new_tokens)
    results = model.fast_generate(prompts, sampling_params=sampling_params, lora_request=None)
    outputs = []
    for r in results:
        try:
            out = r.outputs[0].text
        except Exception:
            out = str(r)
        outputs.append(out)
    return outputs

# --------------------------
# Gold resolution & evaluation saver
# --------------------------
def resolve_gold(ex: Dict[str, Any]):
    # input ex is the processed/formatted row (top-level keys present)
    gold_label = ex.get("answer_label")
    gold_text = ex.get("answer_text")
    raw = ex.get("raw")
    if isinstance(raw, dict):
        if gold_label is None and raw.get("answer_label"):
            gold_label = raw.get("answer_label")
        if gold_text is None and raw.get("answer"):
            gold_text = raw.get("answer")
        if gold_text is None and raw.get("cop") is not None:
            try:
                idx = int(raw.get("cop"))
                opts = raw.get("options") or ex.get("options", [])
                if idx == -1:
                    # explicit no-gold marker
                    pass
                elif 0 <= idx < len(opts):
                    gold_label = chr(ord("A")+idx)
                    gold_text = opts[idx]
                elif 1 <= idx <= len(opts):
                    gold_label = chr(ord("A")+(idx-1))
                    gold_text = opts[idx-1]
            except Exception:
                # non-int cop -> treat as text
                s = str(raw.get("cop")).strip()
                if s:
                    gold_text = gold_text or s
    return gold_label, gold_text

def evaluate_predictions_and_save(rows, markers, preds_map, out_jsonl_path):
    makedirs(os.path.dirname(out_jsonl_path) or ".")
    total = 0
    summary = {}
    for k in preds_map.keys():
        summary[f"{k}_correct"] = 0
    summary["total"] = 0

    with open(out_jsonl_path, "w", encoding="utf-8") as fh:
        for i, ex in enumerate(rows):
            gold_label, gold_text = resolve_gold(ex)
            result = {"id": ex.get("id"), "gold_label": gold_label, "gold_text": gold_text}
            for k, preds in preds_map.items():
                pred_raw = preds[i] if i < len(preds) else ""
                # Save full raw output (reasoning traces included)
                pred_sol = extract_solution_from_text(pred_raw, markers["solution_start"], markers["solution_end"])
                if pred_sol is None:
                    # fallback: look into the head for a letter answer A/B/C...
                    candidate = None
                    head = (pred_raw or "")[:200].upper()
                    # look for "SOLUTION" or single letter patterns
                    if "SOLUTION" in head:
                        # try to find a single-letter after SOLUTION
                        import re
                        m = re.search(r"SOLUTION[^A-Z0-9]*([A-Z])", head)
                        if m:
                            candidate = m.group(1)
                    if candidate is None:
                        for letter in ["A", "B", "C", "D", "E", "F"]:
                            if f"{letter}" in head:
                                candidate = letter
                                break
                    pred_sol = candidate or (pred_raw or "").strip()

                match = False
                if gold_label:
                    if isinstance(pred_sol, str) and pred_sol.strip().upper() == str(gold_label).strip().upper():
                        match = True
                    elif isinstance(pred_sol, str) and str(gold_label).strip().upper() in pred_sol.strip().upper():
                        match = True
                elif gold_text:
                    if isinstance(pred_sol, str) and pred_sol.strip().lower() in str(gold_text).strip().lower():
                        match = True

                # store everything: raw output, extracted solution, match flag
                result[f"pred_{k}"] = pred_raw
                result[f"pred_{k}_solution"] = pred_sol
                result[f"pred_{k}_match"] = match

                if match:
                    summary[f"{k}_correct"] += 1
            summary["total"] += 1
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    for k in preds_map.keys():
        key = f"{k}_acc"
        summary[key] = summary[f"{k}_correct"] / summary["total"] if summary["total"] > 0 else 0.0
    return summary

# --------------------------
# CLI & main (ENTIRE validation enforced)
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline / SFT LoRA / GRPO LoRA on formatted MedMCQA JSONL (single GPU: cuda:0)")
    parser.add_argument("--formatted_jsonl", type=str, required=True, help="Path to the formatted validation JSONL (the ENTIRE file will be used)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--preformat_lora", type=str, default=None)
    parser.add_argument("--grpo_lora", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="experiments")
    parser.add_argument("--prefer_unsloth", action="store_true", help="Prefer unsloth if available")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for transformers generation (tune to GPU memory).")
    args = parser.parse_args()

    makedirs(args.out_dir)
    tokenizer_name = args.tokenizer_name or args.model_name

    print("Using torch device:", torch_device)
    print("Loading tokenizer and building chat template...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    print("Loading formatted dataset (ENTIRE file will be used):", args.formatted_jsonl)
    rows = load_formatted(args.formatted_jsonl)
    if len(rows) == 0:
        print("No rows found in", args.formatted_jsonl)
        sys.exit(1)
    print(f"Loaded {len(rows)} examples (entire validation set).")

    prompts = []
    for r in rows:
        p = apply_chat_template_to_messages(tokenizer, r["messages"], add_generation_prompt=True, tokenize=False)
        prompts.append(p)

    # Baseline
    print("Running baseline generation (no LoRA) on device:", torch_device)
    baseline_preds = None
    try:
        if args.prefer_unsloth and HAS_UNSLOTH:
            print("Using unsloth for baseline generation...")
            baseline_preds = generate_with_unsloth(args.model_name, prompts, lora_path=None, max_new_tokens=args.max_new_tokens)
        else:
            baseline_preds = generate_with_transformers(args.model_name, tokenizer_name, prompts, lora_path=None, max_new_tokens=args.max_new_tokens, device=torch_device, batch_size=args.batch_size)
    except Exception as e:
        print("Baseline generation failed:", e)
        baseline_preds = [""] * len(prompts)

    # SFT LoRA
    sft_preds = [""] * len(prompts)
    if args.preformat_lora and os.path.isdir(args.preformat_lora):
        print("Running generation with preformat LoRA on device:", torch_device)
        if PEFT_AVAILABLE:
            try:
                sft_preds = generate_with_transformers(args.model_name, tokenizer_name, prompts, lora_path=args.preformat_lora, max_new_tokens=args.max_new_tokens, device=torch_device, batch_size=args.batch_size)
            except Exception as e:
                print("SFT LoRA generation failed:", e)
                sft_preds = [""] * len(prompts)
        else:
            print("PEFT not available; skipping SFT LoRA.")
    else:
        print("No preformat LoRA supplied or not found; skipping SFT LoRA step.")

    # GRPO LoRA
    grpo_preds = [""] * len(prompts)
    if args.grpo_lora and os.path.isdir(args.grpo_lora):
        print("Running generation with GRPO LoRA on device:", torch_device)
        if PEFT_AVAILABLE:
            try:
                grpo_preds = generate_with_transformers(args.model_name, tokenizer_name, prompts, lora_path=args.grpo_lora, max_new_tokens=args.max_new_tokens, device=torch_device, batch_size=args.batch_size)
            except Exception as e:
                print("GRPO LoRA generation failed:", e)
                grpo_preds = [""] * len(prompts)
        else:
            print("PEFT not available; skipping GRPO LoRA.")
    else:
        print("No GRPO LoRA supplied or not found; skipping GRPO LoRA step.")

    preds_map = {"baseline": baseline_preds, "sft": sft_preds, "grpo": grpo_preds}
    out_jsonl = os.path.join(args.out_dir, "eval_results.jsonl")
    print("Saving per-example results (including reasoning traces) to:", out_jsonl)
    summary = evaluate_predictions_and_save(rows, markers, preds_map, out_jsonl)
    summary_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("Saved evaluation summary to:", summary_path)
    print("Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

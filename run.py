#!/usr/bin/env python3
"""
run.py — compact pipeline for:
  - downloading MedMCQA (HF: openlifescienceai/medmcqa)
  - building chat template (GRPO / reasoning markers)
  - formatting dataset for SFT/GRPO
  - baseline evaluation with Qwen 4B (unsloth wrapper)
  - SFT pre-finetune to teach formatting (LoRA)
  - GRPO training (LoRA)
  - inference/evaluation

Usage:
  python run.py download
  python run.py prep
  python run.py baseline
  python run.py sft
  python run.py grpo
  python run.py eval
  python run.py all

Make sure to install required packages first (see requirements.txt).
This script expects rewards.py to be present in the same directory.
"""

import os
import sys
import json
import argparse
import pathlib
import time
from typing import List, Dict, Any
import torch
# Force using CUDA:0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = "cuda:0"

# ---- imports that may be heavy ----
try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
# unsloth and vllm/trl imports

try:
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
except Exception:
    SFTTrainer = SFTConfig = GRPOTrainer = GRPOConfig = None
try:
    from vllm import SamplingParams
except Exception:
    SamplingParams = None
from safetensors import safe_open


# Import rewards module (separate file)
try:
    from rewards import (
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    )
except Exception as e:
    print("Could not import rewards.py — please ensure rewards.py exists in the same folder.")
    # We'll continue; grpo won't run until rewards exist.
    match_format_exactly = match_format_approximately = check_answer = check_numbers = None


# --------------------------
# Config block (edit here)
# --------------------------
CONFIG = {
    "model_name": "unsloth/Qwen3-4B-Base",  # default; change to your HF id if needed
    "max_seq_length": 2048,
    "lora_rank": 32,
    "gpu_memory_utilization": 0.9,
    "seed": 3407,
    "data_dir": "data",
    "checkpoints_dir": "checkpoints",
    "experiments_dir": "experiments",
    "preformat_lora_name": "preformat_lora",
    "grpo_lora_name": "grpo_lora",
    "device": DEVICE,
    # SFT hyperparams
    "sft": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 5,
        "num_train_epochs": 2,
        "learning_rate": 2e-4,
        "logging_steps": 5,
    },
    # GRPO hyperparams (small by default; adjust for longer runs)
    "grpo": {
        "temperature": 1.0,
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "num_generations": 4,
        "max_steps": 100,
        "save_steps": 100,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
}


# --------------------------
# Utilities
# --------------------------
def makedirs(*paths):
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# Dataset download + processing
# --------------------------
def download_medmcqa(out_path: str):
    """
    Downloads the Hugging Face MedMCQA dataset (openlifescienceai/medmcqa).
    Normalizes option fields like 'opa','opb','opc','opd' (or a list 'options').
    Detects the correct answer via 'cop' (index), 'answer', 'label', or letter label.
    Saves processed JSONL entries with:
      { 'id', 'question', 'options': [...], 'answer_label': 'C', 'answer_text': 'Atrophy', 'raw': ex }
    """
    makedirs(CONFIG["data_dir"])
    print("Loading dataset: openlifescienceai/medmcqa ... (may take a minute)")
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    out = []
    for i, ex in enumerate(ds):
        q = ex.get("question") or ex.get("Question") or ex.get("prompt") or ""
        # gather option fields robustly (look for opa/opb/... or option1..)
        options = []
        # common keys that might hold options
        # first try an 'options' or 'choices' list
        if ex.get("options") and isinstance(ex.get("options"), (list, tuple)):
            options = [str(x).strip() for x in ex.get("options")]
        elif ex.get("choices") and isinstance(ex.get("choices"), (list, tuple)):
            options = [str(x).strip() for x in ex.get("choices")]
        else:
            # collect keys starting with 'op' followed by letter or digit (opa, opb, opc, op1, op2...)
            op_items = []
            for k, v in ex.items():
                if isinstance(k, str) and k.lower().startswith("op") and v is not None:
                    # record (key, text)
                    op_items.append((k, str(v).strip()))
            # sort by key so opa, opb, opc come in order
            if op_items:
                op_items = sorted(op_items, key=lambda x: x[0])
                options = [t for _, t in op_items]

        # fallback: try option1..option4
        if not options:
            for k in ["option1", "option2", "option3", "option4"]:
                if ex.get(k):
                    options.append(str(ex.get(k)).strip())

        # determine answer index/label/text
        answer_label = None
        answer_text = None
        # possible answer fields
        possible_ans = None
        for cand in ["cop", "answer", "label", "ans", "correct", "correct_option"]:
            if cand in ex and ex[cand] not in (None, ""):
                possible_ans = ex[cand]
                break

        if possible_ans is not None:
            # If it's numeric index:
            try:
                idx = int(possible_ans)
                # handle 1-based or 0-based indices
                if 0 <= idx < len(options):
                    chosen_idx = idx
                elif 1 <= idx <= len(options):
                    chosen_idx = idx - 1
                else:
                    chosen_idx = None
                if chosen_idx is not None:
                    answer_text = options[chosen_idx]
                    answer_label = chr(ord("A") + chosen_idx)
            except Exception:
                # if it's a letter like 'A' or 'c'
                s = str(possible_ans).strip()
                if len(s) == 1 and s.isalpha():
                    letter = s.upper()
                    idx = ord(letter) - ord("A")
                    if 0 <= idx < len(options):
                        answer_label = letter
                        answer_text = options[idx]
                else:
                    # maybe the dataset already contains the answer text; try to match to one of options
                    s = s.strip()
                    for j, opt in enumerate(options):
                        if s.lower() == opt.lower() or s in opt:
                            answer_text = opt
                            answer_label = chr(ord("A") + j)
                            break
                    # if still none, store the raw string as answer_text
                    if answer_text is None:
                        answer_text = s
                        answer_label = None

        out.append({
            "id": i,
            "question": q,
            "options": options,
            "answer_label": answer_label,   # e.g., "C"
            "answer_text": answer_text,     # e.g., "Atrophy"
            "raw": ex,
        })

    # Save to JSONL
    makedirs(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")
    print(f"Saved processed MedMCQA to {out_path} (n={len(out)})")
    return out


# --------------------------
# Chat template + tokenizer helpers
# --------------------------
def build_chat_template(tokenizer):
    """
    Build the jinja chat template for reasoning traces and apply to tokenizer.
    Uses markers like:
      reasoning_start = "<start_working_out>"
      reasoning_end   = "<end_working_out>"
      solution_start  = "<SOLUTION>"
      solution_end    = "</SOLUTION>"
    """
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

    # apply to huggingface tokenizer if supports chat_template attribute (unsloth tokenizer does)
    try:
        tokenizer.chat_template = chat_template
        print("Applied custom chat_template to tokenizer.")
    except Exception:
        # fallback: store template in tokenizer object
        setattr(tokenizer, "chat_template", chat_template)
        print("Stored chat_template attribute on tokenizer (fallback).")

    # return markers for downstream
    return {
        "system_prompt": system_prompt,
        "reasoning_start": reasoning_start,
        "reasoning_end": reasoning_end,
        "solution_start": solution_start,
        "solution_end": solution_end,
    }


def apply_chat_template_to_messages(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False):
    """
    Helper wrapper similar to tokenizer.apply_chat_template used in Unslo th.
    Behavior:
      - If the tokenizer has attribute `apply_chat_template` AND tokenizer.chat_template is set: use it.
      - Otherwise fall back to a manual concatenation rendering and optionally tokenize.
    This avoids calling tokenizer.apply_chat_template when tokenizer.chat_template is not set,
    which causes transformers to raise a ValueError.
    """
    # If tokenizer has a direct method and chat_template is already set, use it safely
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize)
        except Exception as e:
            # fall through to manual fallback if tokenizer.apply_chat_template unexpectedly fails
            print("Warning: tokenizer.apply_chat_template failed, falling back to manual rendering:", e)

    # Manual fallback: produce a simple concatenated prompt with system prompt + user messages + generation marker
    # Try to use tokenizer.chat_template content if present; otherwise build a simple default system prompt.
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        # create a concise system prompt that mirrors the main script's format
        system_prompt = (
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between <start_working_out> and <end_working_out>.\n"
            "Then, provide your solution between <SOLUTION></SOLUTION>"
        )
    else:
        # Extract the system_prompt from the template if possible (best-effort),
        # otherwise fall back to the same concise system prompt.
        try:
            # crude extraction: find the literal between "{{ '" and "' + eos_token"
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

    # Build the concatenated text (system prompt + loop messages)
    pieces = []
    if messages and messages[0].get("role") == "system":
        pieces.append(messages[0]["content"])
        loop_messages = messages[1:]
    else:
        pieces.append(system_prompt)
        loop_messages = messages

    for m in loop_messages:
        # For assistant messages include eos-like newline to separate from next items
        pieces.append(m.get("content", ""))

    if add_generation_prompt:
        pieces.append("<start_working_out>")

    raw = "\n".join(pieces)

    if tokenize:
        # Use tokenizer.encode / return_tensors path depending on tokenizer capabilities
        try:
            enc = tokenizer(raw, return_tensors="pt")
            # return token ids list (match earlier code's expected shape)
            # If caller expects a list-of-ids, return that; else return raw enc dict
            if "input_ids" in enc:
                return enc["input_ids"][0].tolist()
            return enc
        except Exception as e:
            # if tokenization fails, return the raw string so higher-level code can handle it
            print("Warning: tokenizer() failed in fallback tokenize step:", e)
            return raw

    return raw


# --------------------------
# Data formatting for SFT/GRPO
# --------------------------
def format_medmcqa_for_chat(in_jsonl: str, out_jsonl: str, tokenizer, markers: Dict[str, str], subset_limit: int = None):
    """
    Convert processed medmcqa jsonl into chat-format strings for SFT/GRPO.
    Prompts explicitly include choices and request the MODEL to answer with the OPTION LABEL
    (e.g., A, B, C) between <SOLUTION> and </SOLUTION>.
    """
    makedirs(os.path.dirname(out_jsonl) or ".")
    system_prompt = markers["system_prompt"]
    reasoning_start = markers["reasoning_start"]
    reasoning_end = markers["reasoning_end"]
    solution_start = markers["solution_start"]
    solution_end = markers["solution_end"]

    out_lines = []
    with open(in_jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if subset_limit and i >= subset_limit:
                break
            ex = json.loads(line)
            q = ex["question"]
            options = ex.get("options") or []
            # Format choices with labels A., B., etc.
            if options:
                labeled = [f"{chr(ord('A')+idx)}. {opt}" for idx, opt in enumerate(options)]
                choices_text = "\n".join(labeled)
                prompt_text = (
                    f"{q}\n\nOptions:\n{choices_text}\n\n"
                    f"Please answer with the OPTION LABEL only (for example 'A' or 'B').\n"
                    f"Place the final option label between {solution_start} and {solution_end}."
                )
            else:
                prompt_text = (
                    f"{q}\n\nPlease place the final answer between {solution_start} and {solution_end}."
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]

            # If we have a gold label, include an assistant target using the labeling convention
            answer_label = ex.get("answer_label")
            answer_text = ex.get("answer_text")
            if answer_label is not None:
                assistant_content = (
                    f"{reasoning_start}Working...{reasoning_end}"
                    f"{solution_start}{answer_label}{solution_end}"
                )
                messages.append({"role": "assistant", "content": assistant_content})
            elif answer_text is not None and options:
                # attempt to find label from answer_text
                label = None
                for j, opt in enumerate(options):
                    if answer_text.strip().lower() == opt.strip().lower() or answer_text.strip() in opt:
                        label = chr(ord("A")+j)
                        break
                if label is not None:
                    assistant_content = (
                        f"{reasoning_start}Working...{reasoning_end}"
                        f"{solution_start}{label}{solution_end}"
                    )
                    messages.append({"role": "assistant", "content": assistant_content})
                else:
                    # No reliable label -> include text answer in SOLUTION tags (less ideal)
                    assistant_content = (
                        f"{reasoning_start}Working...{reasoning_end}"
                        f"{solution_start}{answer_text}{solution_end}"
                    )
                    messages.append({"role": "assistant", "content": assistant_content})

            out_lines.append({"id": ex.get("id"), "messages": messages, "raw": ex})

    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in out_lines:
            fh.write(json.dumps(row) + "\n")
    print(f"Wrote formatted dataset to {out_jsonl} (n={len(out_lines)})")
    return out_lines


# --------------------------
# Baseline evaluation
# --------------------------
def baseline_evaluate_model(model_name: str, formatted_jsonl: str, tokenizer_name: str = None, max_new_tokens: int = 64, limit: int = 100):
    """
    Improved baseline evaluation:
     - Reads gold label from answer_label / answer_text or falls back to raw.answer/cop.
     - Uses tokenizer + transformers.generate in a safe way when FastLanguageModel not available.
     - Tries to extract <SOLUTION>... </SOLUTION> from the generation and compare to gold label/text.
    """
    print("Starting baseline evaluation...")
    # load data
    examples = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            examples.append(json.loads(line))
    if len(examples) == 0:
        print("No examples found.")
        return

    # load tokenizer
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    # load model (prefer unsloth FastLanguageModel if available)
    model = None
    use_unsloth = False
    if FastLanguageModel is not None:
        try:
            print("Loading model via Unslo th FastLanguageModel (preferred) ...")
            model = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=CONFIG["max_seq_length"],
                load_in_4bit=False,
                fast_inference=True,
                max_lora_rank=CONFIG["lora_rank"],
                gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
            )
            use_unsloth = True
        except Exception as e:
            print("FastLanguageModel load failed, falling back to transformers:", e)
            model = None

    if model is None:
        print("Using transformers AutoModelForCausalLM (may be slow / large).")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name).to(CONFIG["device"])

    correct = 0
    total = 0

    for ex in examples:
        messages = ex["messages"]
        # Must add generation prompt
        text = apply_chat_template_to_messages(tokenizer, messages, add_generation_prompt=True, tokenize=False)

        out = ""
        try:
            if use_unsloth and hasattr(model, "fast_generate"):
                # Unslo th fast_generate expects a list of prompts
                sampling_params = SamplingParams(temperature=1.0, max_tokens=max_new_tokens)
                out = model.fast_generate(
                    [text],
                    sampling_params=sampling_params,
                    lora_request=None,
                )[0].outputs[0].text
            else:
                # transformers generate path (robust)
                inputs = tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(CONFIG["device"])
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(CONFIG["device"])

                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
                )

                # decode whole sequence, then remove the prompt portion to obtain only the new tokens
                decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                # decode the prompt tokens to find prompt length in chars
                prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                # If decoded starts with prompt_decoded, strip it to keep only model-generated suffix
                if decoded.startswith(prompt_decoded):
                    out = decoded[len(prompt_decoded):].strip()
                else:
                    out = decoded

        except Exception as e:
            print("Generation error (continuing):", e)
            out = ""

        # extract solution between tags
        sol = extract_solution_from_text(out, markers["solution_start"], markers["solution_end"])
        # Determine gold answer robustly:
        # Look for top-level 'answer_label' or 'answer_text' fields, then 'raw' nested fields (some entries nest the processed row)
        gold_label = None
        gold_text = None
        if isinstance(ex.get("raw"), dict):
            r = ex["raw"]
            # sometimes the processed answer fields are nested inside r['raw'] again; test both
            if "answer_label" in r and r.get("answer_label"):
                gold_label = r.get("answer_label")
            if "answer_text" in r and r.get("answer_text"):
                gold_text = r.get("answer_text")
            # fallback to top-level processed fields in 'ex' if present
        if gold_label is None and "answer_label" in ex:
            gold_label = ex.get("answer_label")
        if gold_text is None and "answer_text" in ex:
            gold_text = ex.get("answer_text")
        # final fallback: older fields
        if gold_text is None:
            # sometimes dataset uses 'answer', 'cop'
            if isinstance(ex.get("raw"), dict) and ex["raw"].get("answer"):
                gold_text = ex["raw"].get("answer")
            if isinstance(ex.get("raw"), dict) and ex["raw"].get("cop") is not None:
                try:
                    idx = int(ex["raw"].get("cop"))
                    # cop sometimes 0-based or 1-based — choose 0-based if within range
                    opts = ex["raw"].get("options") or ex.get("options", [])
                    if 0 <= idx < len(opts):
                        gold_label = chr(ord("A") + idx)
                        gold_text = opts[idx]
                    elif 1 <= idx <= len(opts):
                        gold_label = chr(ord("A") + (idx-1))
                        gold_text = opts[idx-1]
                except Exception:
                    pass

        # Determine match:
        match = False
        if gold_label is not None:
            # If solution is label directly (A/B/C), compare labels
            if sol is not None and sol.strip().upper() == str(gold_label).strip().upper():
                match = True
            # Also accept if sol contains the gold label (e.g., "Answer: C")
            elif sol is not None and str(gold_label).strip().upper() in sol.strip().upper():
                match = True
            else:
                match = False
        elif gold_text is not None:
            # Compare by text containment (case-insensitive)
            if sol is not None and sol.strip().lower() in str(gold_text).strip().lower():
                match = True
            else:
                match = False
        else:
            match = False

        print(f"[{total+1}] gold_label={gold_label} gold_text={gold_text} | pred={sol} | match={match}")

        if match:
            correct += 1
        total += 1

    print(f"Baseline: {correct}/{total} = {correct/total if total>0 else 0:.4f}")


def extract_solution_from_text(text: str, sol_start: str, sol_end: str):
    # naive extraction of the first occurrence
    s = text.find(sol_start)
    if s == -1:
        return None
    s2 = text.find(sol_end, s + len(sol_start))
    if s2 == -1:
        return text[s + len(sol_start):].strip()
    return text[s + len(sol_start):s2].strip()


# --------------------------
# SFT pre-finetune (small formatting priming)
# --------------------------
def run_sft_pretrain(formatted_jsonl: str, model_name: str, out_lora_dir: str):
    """
    Fallback SFT using transformers + peft (if Unslo th FastLanguageModel is not installed).
    Saves a LoRA adapter directory at checkpoints/<out_lora_dir>.
    """
    print("Preparing SFT dataset from:", formatted_jsonl)
    rows = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rows.append(json.loads(line))

    # Build 'text' examples: use the messages and include assistant target if present.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    texts = []
    for r in rows:
        messages = r["messages"]
        # If assistant present, include assistant content as training target.
        # We'll simply serialize messages via apply_chat_template (fallback) and use that string.
        full = apply_chat_template_to_messages(tokenizer, messages, add_generation_prompt=False, tokenize=False)
        texts.append({"text": full})

    # Convert to HF dataset
    hf_ds = Dataset.from_list(texts)

    # Tokenize
    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=CONFIG["max_seq_length"])
        out["labels"] = out["input_ids"].copy()  # causal LM labels == input_ids
        return out

    tokenized = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Load base model (transformers)
    import transformers
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("Loading base model via transformers:", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # device_map auto if you have accelerate/torch >=
    # If memory is tight, we could load with load_in_8bit via bitsandbytes; for now default.

    # Prepare model for LoRA (and possible 8-bit)
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # prepare_model_for_kbit_training optional; continue if not available
        pass

    # Configure LoRA
    lora_rank = CONFIG["lora_rank"]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Small TrainingArguments for quick SFT
    train_args = TrainingArguments(
        output_dir=os.path.join(CONFIG["checkpoints_dir"], out_lora_dir),
        per_device_train_batch_size=CONFIG["sft"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["sft"]["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["sft"]["num_train_epochs"],
        learning_rate=CONFIG["sft"]["learning_rate"],
        logging_steps=CONFIG["sft"]["logging_steps"],
        fp16=True if torch.cuda.is_available() else False,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # Define data collator for causal LM
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting SFT pre-finetune (transformers+peft)...")
    trainer.train()

    # Save LoRA weights (PEFT save)
    makedirs(CONFIG["checkpoints_dir"])
    out_path = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir)
    print("Saving LoRA to", out_path)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print("Saved preformat LoRA to", out_path)
    return out_path


# --------------------------
# GRPO training
# --------------------------
def run_grpo_training(formatted_jsonl: str, model_name: str, preformat_lora_path: str = None, out_lora_dir: str = None):
    """
    Run GRPO training using trl.GRPOTrainer and the reward functions imported from rewards.py
    """
    if GRPOTrainer is None or SamplingParams is None:
        raise RuntimeError("trl.GRPOTrainer or vllm.SamplingParams not available. Install required libs.")

    print("Loading formatted dataset...")
    rows = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    # Convert to HF Dataset
    prompts = []
    answers = []
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    for r in rows:
        messages = r["messages"]
        # For generation, we will use chat_template with add_generation_prompt True
        text = apply_chat_template_to_messages(tokenizer, messages, add_generation_prompt=True, tokenize=False)
        prompts.append(text)
        # collect answer if present
        raw_ans = r["raw"].get("answer")
        answers.append(raw_ans if raw_ans is not None else "")

    # Build vllm sampling params
    sampling = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=CONFIG["seed"],
        stop=[tokenizer.eos_token] if hasattr(tokenizer, "eos_token") else None,
        include_stop_str_in_output=True,
    )

    # Load model via FastLanguageModel
    print("Loading model (FastLanguageModel)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=CONFIG["lora_rank"],
        gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
    )

    # Apply PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=CONFIG["lora_rank"] * 2,
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
    )

    # Optionally load preformat LoRA
    if preformat_lora_path:
        try:
            print("Loading preformat LoRA:", preformat_lora_path)
            model.load_lora(preformat_lora_path)
        except Exception as e:
            print("Could not load preformat LoRA:", e)

    # GRPO config (mirror snippet)
    grpo_args = GRPOConfig(
        vllm_sampling_params=sampling,
        temperature=CONFIG["grpo"]["temperature"],
        learning_rate=CONFIG["grpo"]["learning_rate"],
        weight_decay=CONFIG["grpo"]["weight_decay"],
        warmup_ratio=CONFIG["grpo"]["warmup_ratio"],
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=CONFIG["grpo"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["grpo"]["gradient_accumulation_steps"],
        num_generations=CONFIG["grpo"]["num_generations"],
        max_prompt_length=512,  # will be adjusted below based on token lengths
        max_completion_length=CONFIG["max_seq_length"] - 512,
        max_steps=CONFIG["grpo"]["max_steps"],
        save_steps=CONFIG["grpo"]["save_steps"],
        report_to="none",
        output_dir=CONFIG["checkpoints_dir"],
    )

    # adjust max_prompt_length by computing a quantile of prompt token lengths:
    print("Tokenizing prompts to compute prompt length quantile...")
    token_lens = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt")["input_ids"][0]
        token_lens.append(ids.shape[0])
    import numpy as np
    max_prompt_length = int(np.quantile(np.array(token_lens), 0.9))
    grpo_args.max_prompt_length = max_prompt_length + 1
    grpo_args.max_completion_length = CONFIG["max_seq_length"] - grpo_args.max_prompt_length
    print(f"Set max_prompt_length={grpo_args.max_prompt_length} max_completion_length={grpo_args.max_completion_length}")

    # Build a minimal train_dataset from prompts + answers for GRPOTrainer
    train_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

    # Reward functions: ensure they are present
    reward_funcs = []
    if match_format_exactly: reward_funcs.append(match_format_exactly)
    if match_format_approximately: reward_funcs.append(match_format_approximately)
    if check_answer: reward_funcs.append(check_answer)
    if check_numbers: reward_funcs.append(check_numbers)
    if not reward_funcs:
        raise RuntimeError("No reward functions available. Ensure rewards.py is present and imports succeed.")

    # Create GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
    )

    print("Starting GRPO training (this may take a long time)...")
    trainer.train()

    # Save LoRA adapter after training
    makedirs(CONFIG["checkpoints_dir"])
    out_dir = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir or CONFIG["grpo_lora_name"])
    model.save_lora(out_dir)
    print("Saved GRPO LoRA to", out_dir)
    return out_dir


# --------------------------
# Inference with saved LoRA
# --------------------------
def run_inference_with_lora(model_name: str, tokenizer_name: str, lora_path: str, prompts: List[str], max_new_tokens: int = 256):
    """
    Inference using Transformers + PEFT (works when Unslo th is not installed).
    - Loads base model via transformers (device_map='auto' if available).
    - Loads LoRA adapter using PeftModel.from_pretrained().
    - Generates outputs and returns list of generated strings (only the newly generated suffix).
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import torch

    print("Loading tokenizer and base model for inference (transformers + peft)...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    # Try to load base model in float16 and with device_map auto if available to reduce RAM.
    try:
        # prefer low_cpu_mem_usage + device_map if HF supports it
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print("Warning: couldn't use device_map/float16 shortcut, falling back to simple load:", e)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(CONFIG["device"])

    # Attach LoRA adapter if available
    if lora_path and os.path.isdir(lora_path):
        try:
            model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
            print("Loaded LoRA adapter from", lora_path)
        except Exception as e:
            print("Failed to load LoRA adapter via PeftModel:", e)
            print("Continuing with base model (no adapter).")
    else:
        print("No LoRA adapter found at", lora_path, "- continuing with base model.")

    model.eval()
    outputs = []
    for p in prompts:
        # tokenize and move to device
        inputs = tokenizer(p, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
            )

        # decode full sequence then remove prompt portion to get only generated suffix
        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        if decoded.startswith(prompt_decoded):
            out = decoded[len(prompt_decoded):].strip()
        else:
            out = decoded.strip()
        outputs.append(out)

    return outputs


# --------------------------
# CLI and orchestration
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Compact MedMCQA -> GRPO pipeline (single-file)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="Download and process MedMCQA dataset")
    sub.add_parser("prep", help="Build chat template and format dataset for SFT/GRPO")
    sub.add_parser("baseline", help="Run baseline evaluation (generate answers with base model)")
    sub.add_parser("sft", help="Run SFT pre-finetune to teach formatting")
    sub.add_parser("grpo", help="Run GRPO training")
    sub.add_parser("eval", help="Run inference using saved LoRA")
    sub.add_parser("all", help="Run entire pipeline in sequence (download->prep->baseline->sft->grpo->eval)")

    parser.add_argument("--limit", type=int, default=200, help="limit number of examples to process (for quick tests)")
    parser.add_argument("--model_name", type=str, default=CONFIG["model_name"])
    parser.add_argument("--formatted_jsonl", type=str, default=os.path.join(CONFIG["data_dir"], "medmcqa_formatted.jsonl"))
    parser.add_argument("--processed_jsonl", type=str, default=os.path.join(CONFIG["data_dir"], "medmcqa_processed.jsonl"))
    parser.add_argument("--preformat_lora", type=str, default=os.path.join(CONFIG["checkpoints_dir"], CONFIG["preformat_lora_name"]))
    parser.add_argument("--grpo_lora", type=str, default=os.path.join(CONFIG["checkpoints_dir"], CONFIG["grpo_lora_name"]))
    args = parser.parse_args()

    makedirs(CONFIG["data_dir"], CONFIG["checkpoints_dir"], CONFIG["experiments_dir"])

    if args.cmd == "download":
        download_medmcqa(args.processed_jsonl)

    elif args.cmd == "prep":
        # build tokenizer & template
        print("Loading tokenizer to build chat template...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        # format dataset for SFT/GRPO
        format_medmcqa_for_chat(args.processed_jsonl, args.formatted_jsonl, tokenizer, markers, subset_limit=args.limit)

    elif args.cmd == "baseline":
        baseline_evaluate_model(args.model_name, args.formatted_jsonl, tokenizer_name=args.model_name, limit=args.limit)

    elif args.cmd == "sft":
        out = run_sft_pretrain(args.formatted_jsonl, args.model_name, CONFIG["preformat_lora_name"])
        print("SFT pretrain done, saved:", out)

    elif args.cmd == "grpo":
        out = run_grpo_training(args.formatted_jsonl, args.model_name, preformat_lora_path=args.preformat_lora, out_lora_dir=CONFIG["grpo_lora_name"])
        print("GRPO done, saved:", out)

    elif args.cmd == "eval":
        # run a few interactive prompts with the grpo lora
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        # load a few prompts from formatted_jsonl
        prompts = []
        with open(args.formatted_jsonl, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 10:
                    break
                row = json.loads(line)
                text = apply_chat_template_to_messages(tokenizer, row["messages"], add_generation_prompt=True, tokenize=False)
                prompts.append(text)
        outputs = run_inference_with_lora(args.model_name, args.model_name, args.grpo_lora, prompts, max_new_tokens=256)
        for i, out in enumerate(outputs):
            sol = extract_solution_from_text(out, markers["solution_start"], markers["solution_end"])
            print(f"=== Example {i} ===\nGen:\n{out}\nExtracted solution: {sol}\n")

    elif args.cmd == "all":
        # run everything in sequence with small limits for quick test
        download_medmcqa(args.processed_jsonl)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        format_medmcqa_for_chat(args.processed_jsonl, args.formatted_jsonl, tokenizer, markers, subset_limit=args.limit)
        baseline_evaluate_model(args.model_name, args.formatted_jsonl, tokenizer_name=args.model_name, limit=min(50, args.limit))
        run_sft_pretrain(args.formatted_jsonl, args.model_name, CONFIG["preformat_lora_name"])
        run_grpo_training(args.formatted_jsonl, args.model_name, preformat_lora_path=os.path.join(CONFIG["checkpoints_dir"], CONFIG["preformat_lora_name"]))
        print("Pipeline complete. Inspect checkpoints/ and experiments/ for outputs.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

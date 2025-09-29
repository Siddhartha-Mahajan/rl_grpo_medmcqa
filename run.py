#!/usr/bin/env python3
"""
run.py — compact pipeline for MedMCQA with validation used for all testing.
  - download: write processed JSONL per split (train/validation/test if present)
  - prep: format train + validation (ignore test)
  - baseline: evaluate on validation split (batched, labeled examples only)
  - sft: SFT pre-finetune on train split (LoRA)
  - grpo: GRPO training on train split (LoRA), then quick eval on validation
  - eval: run inference on a handful of validation prompts using saved LoRA

Usage:
  python run.py download
  python run.py prep
  python run.py baseline
  python run.py sft
  python run.py grpo
  python run.py eval
  python run.py all

Notes:
 - This script purposely uses the validation split for evaluation and ignores test.
 - Ensure rewards.py exists for GRPO.
"""
import os
import json
import argparse
import pathlib
from typing import List, Dict
import torch

# force a visible GPU device if not set externally
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = "cuda:0"

# optional heavy libs guarded
try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

try:
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
except Exception:
    SFTTrainer = SFTConfig = GRPOTrainer = GRPOConfig = None

try:
    from vllm import SamplingParams
except Exception:
    SamplingParams = None

# rewards.py expected in same folder
try:
    from rewards import (
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    )
except Exception:
    match_format_exactly = match_format_approximately = check_answer = check_numbers = None
    print("Warning: rewards.py not found or failed to import — GRPO will not run until it's fixed.")

# --------------------------
# Config
# --------------------------
CONFIG = {
    "model_name": "unsloth/Qwen3-4B-Base",
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
    "sft": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 5,
        "num_train_epochs": 2,
        "learning_rate": 2e-4,
        "logging_steps": 5,
    },
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
    "eval_batch_size": 16,
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
# Download + process (per-split)
# --------------------------
def download_medmcqa(out_dir: str):
    """
    Download MedMCQA and write processed JSONL per split (whatever load_dataset returns).
    Outputs files medmcqa_processed.<split>.jsonl in out_dir.
    """
    makedirs(out_dir)
    print("Loading dataset: openlifescienceai/medmcqa ...")
    ds_all = load_dataset("openlifescienceai/medmcqa")
    saved = {}
    for split in ds_all.keys():
        ds = ds_all[split]
        out = []
        for i, ex in enumerate(ds):
            q = ex.get("question") or ex.get("Question") or ex.get("prompt") or ""
            # gather options robustly
            options = []
            if ex.get("options") and isinstance(ex.get("options"), (list, tuple)):
                options = [str(x).strip() for x in ex.get("options")]
            elif ex.get("choices") and isinstance(ex.get("choices"), (list, tuple)):
                options = [str(x).strip() for x in ex.get("choices")]
            else:
                op_items = []
                for k, v in ex.items():
                    if isinstance(k, str) and k.lower().startswith("op") and v is not None:
                        op_items.append((k, str(v).strip()))
                if op_items:
                    op_items = sorted(op_items, key=lambda x: x[0])
                    options = [t for _, t in op_items]
            if not options:
                for k in ["option1", "option2", "option3", "option4"]:
                    if ex.get(k):
                        options.append(str(ex.get(k)).strip())

            answer_label = None
            answer_text = None
            possible_ans = None
            for cand in ["cop", "answer", "label", "ans", "correct", "correct_option"]:
                if cand in ex and ex[cand] not in (None, ""):
                    possible_ans = ex[cand]
                    break

            if possible_ans is not None:
                try:
                    idx = int(possible_ans)
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
                    s = str(possible_ans).strip()
                    if len(s) == 1 and s.isalpha():
                        letter = s.upper()
                        idx = ord(letter) - ord("A")
                        if 0 <= idx < len(options):
                            answer_label = letter
                            answer_text = options[idx]
                    else:
                        s = s.strip()
                        for j, opt in enumerate(options):
                            if s.lower() == opt.lower() or s in opt:
                                answer_text = opt
                                answer_label = chr(ord("A") + j)
                                break
                        if answer_text is None:
                            answer_text = s
                            answer_label = None

            out.append({
                "id": i,
                "question": q,
                "options": options,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "raw": ex,
            })

        out_path = os.path.join(out_dir, f"medmcqa_processed.{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in out:
                fh.write(json.dumps(row) + "\n")
        print(f"Saved processed split '{split}' to {out_path} (n={len(out)})")
        saved[split] = out_path
    return saved


# --------------------------
# Chat template + helpers
# --------------------------
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
    # Prefer tokenizer.apply_chat_template if available, otherwise manual
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize)
        except Exception:
            pass

    # manual fallback
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
# Format dataset to chat-style (train + validation)
# --------------------------
def format_medmcqa_for_chat(in_jsonl: str, out_jsonl: str, tokenizer, markers: Dict[str, str], subset_limit: int = None):
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

            answer_label = ex.get("answer_label")
            answer_text = ex.get("answer_text")
            if answer_label is not None:
                assistant_content = (
                    f"{reasoning_start}Working...{reasoning_end}"
                    f"{solution_start}{answer_label}{solution_end}"
                )
                messages.append({"role": "assistant", "content": assistant_content})
            elif answer_text is not None and options:
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
# Helpers: extraction fallback
# --------------------------
import re

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


def extract_pred_with_fallback(out: str, sol_start: str, sol_end: str, options: List[str] = None):
    # 1) tags
    sol = extract_solution_from_text(out, sol_start, sol_end)
    if sol:
        return sol.strip()
    if not out:
        return None
    # 2) "Answer: C" style
    m2 = re.search(r'(answer|ans|final)\s*[:\-]?\s*([A-D])\b', out, re.IGNORECASE)
    if m2:
        return m2.group(2).upper()
    # 3) single letter A-D somewhere
    m = re.search(r'\b([A-D])\b', out, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 4) match option text if options provided
    if options:
        out_low = out.lower()
        for idx, opt in enumerate(options):
            if opt and opt.lower() in out_low:
                return chr(ord("A") + idx)
    return None


# --------------------------
# Baseline evaluation on validation (batched; labeled only)
# --------------------------
def baseline_evaluate_model(model_name: str, formatted_jsonl: str, tokenizer_name: str = None,
                            max_new_tokens: int = 64, limit: int = 100, batch_size: int = None):
    """
    Baseline evaluation on the formatted JSONL (validation split in your pipeline).
    This version *robustly* detects gold labels from multiple possible fields
    (answer_label/answer_text at top-level, or inside raw; handles cop indices).
    """
    import math
    print("Starting baseline evaluation (robust label detection, batched)...")

    # load rows (respect limit: if limit is None or <=0, load everything)
    raw_rows = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and limit > 0 and i >= limit:
                break
            raw_rows.append(json.loads(line))
    if not raw_rows:
        print("No examples loaded from", formatted_jsonl)
        return

    # helper: check whether an example has a usable gold (label or text or cop)
    def get_gold_from_example(ex):
        # return tuple (gold_label_or_none, gold_text_or_none)
        # check top-level processed fields first
        gl = ex.get("answer_label")
        gt = ex.get("answer_text")
        # then check raw dict
        r = ex.get("raw") or {}
        if (gl is None or gl == "") and isinstance(r, dict):
            # some processed rows keep gold inside raw
            if r.get("answer_label"):
                gl = r.get("answer_label")
            # sometimes gold text stored in 'answer' or 'answer_text'
            if gt is None or gt == "":
                if r.get("answer_text"):
                    gt = r.get("answer_text")
                elif r.get("answer"):
                    gt = r.get("answer")
            # cop index (could be 0 or 1-based, -1 meaning no label)
            cop = r.get("cop", None)
            if (gl is None or gl == "") and cop is not None and cop not in ("", None):
                try:
                    idx = int(cop)
                    opts = r.get("options") or ex.get("options") or []
                    # treat -1 as unlabeled
                    if idx == -1:
                        pass
                    elif 0 <= idx < len(opts):
                        gl = chr(ord("A") + idx)
                        gt = gt or (opts[idx] if idx < len(opts) else None)
                    elif 1 <= idx <= len(opts):
                        # 1-based index
                        gl = chr(ord("A") + (idx - 1))
                        gt = gt or (opts[idx - 1] if idx - 1 < len(opts) else None)
                except Exception:
                    # non-integer cop could be text (treat as gold text)
                    s = str(cop).strip()
                    if s:
                        gt = gt or s

        # normalize empty strings -> None
        if isinstance(gl, str) and gl.strip() == "":
            gl = None
        if isinstance(gt, str) and gt.strip() == "":
            gt = None
        return gl, gt

    # split labeled vs unlabeled
    labeled = []
    unlabeled = []
    for ex in raw_rows:
        gl, gt = get_gold_from_example(ex)
        if gl is not None or gt is not None:
            # attach discovered gold to the example for reuse
            ex["_gold_label"] = gl
            ex["_gold_text"] = gt
            labeled.append(ex)
        else:
            unlabeled.append(ex)

    print(f"Loaded {len(raw_rows)} examples -> {len(labeled)} labeled, {len(unlabeled)} unlabeled (unlabeled will be skipped).")
    if len(labeled) == 0:
        print("No labeled examples found — cannot compute baseline accuracy. Check processed files or use a different split (train/validation).")
        # show a small sample for debugging
        if len(raw_rows) > 0:
            print("Sample processed row (first):")
            import pprint
            pprint.pprint(raw_rows[0])
        return

    # prepare tokenizer + model
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    model = None
    use_unsloth = False
    if FastLanguageModel is not None:
        try:
            print("Attempting to load model via Unsloth FastLanguageModel ...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=CONFIG["max_seq_length"],
                load_in_4bit=False,
                fast_inference=True,
                max_lora_rank=CONFIG["lora_rank"],
                gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
            )
            use_unsloth = True
        except Exception as e:
            print("Unsloth load failed, falling back to transformers:", e)
            model = None

    if model is None:
        print("Loading base model via transformers (may be slower)...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name).to(CONFIG["device"])

    if batch_size is None:
        batch_size = CONFIG.get("eval_batch_size", 8)

    # build prompts for labeled examples
    prompts = []
    ex_list = []
    for ex in labeled:
        text = apply_chat_template_to_messages(tokenizer, ex["messages"], add_generation_prompt=True, tokenize=False)
        prompts.append(text)
        ex_list.append(ex)

    correct = 0
    total = 0

    # batched generation / eval
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_exs = ex_list[start:start + batch_size]
        out_texts = [""] * len(batch_prompts)

        try:
            if use_unsloth and hasattr(model, "fast_generate"):
                sampling_params = SamplingParams(temperature=1.0, max_tokens=max_new_tokens)
                results = model.fast_generate(batch_prompts, sampling_params=sampling_params, lora_request=None)
                for i, r in enumerate(results):
                    try:
                        out_texts[i] = r.outputs[0].text
                    except Exception:
                        out_texts[i] = getattr(r, "text", "")
            else:
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["max_seq_length"])
                input_ids = enc["input_ids"].to(CONFIG["device"])
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(CONFIG["device"])

                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
                )

                for i in range(len(batch_prompts)):
                    decoded = tokenizer.decode(gen_ids[i], skip_special_tokens=False)
                    prompt_decoded = tokenizer.decode(enc["input_ids"][i], skip_special_tokens=False)
                    if decoded.startswith(prompt_decoded):
                        out_texts[i] = decoded[len(prompt_decoded):].strip()
                    else:
                        out_texts[i] = decoded
        except Exception as e:
            print("Batch generation failed at start", start, " — falling back to per-example. Exception:", e)
            for i, p in enumerate(batch_prompts):
                try:
                    inputs = tokenizer(p, return_tensors="pt").to(CONFIG["device"])
                    gen_ids = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
                    decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
                    out_texts[i] = decoded[len(prompt_decoded):].strip() if decoded.startswith(prompt_decoded) else decoded
                except Exception as e2:
                    print("Per-example fallback failed:", e2)
                    out_texts[i] = ""

        # evaluate outputs
        for i, out in enumerate(out_texts):
            ex = batch_exs[i]
            options = ex.get("raw", {}).get("options") or ex.get("options") or None
            pred = extract_pred_with_fallback(out, markers["solution_start"], markers["solution_end"], options=options)

            # use gold we precomputed
            gold_label = ex.get("_gold_label")
            gold_text = ex.get("_gold_text")

            match = False
            if gold_label is not None:
                if pred is not None and str(pred).strip().upper() == str(gold_label).strip().upper():
                    match = True
                elif pred is not None and str(gold_label).strip().upper() in str(pred).strip().upper():
                    match = True
            elif gold_text is not None:
                if pred is not None and str(pred).strip().lower() in str(gold_text).strip().lower():
                    match = True

            print(f"[{total+1}] gold_label={gold_label} gold_text={gold_text} | pred={pred} | match={match}")
            if match:
                correct += 1
            total += 1

    print(f"Baseline: {correct}/{total} = {correct/total if total>0 else 0:.4f}")


# --------------------------
# SFT pre-finetune (train split)
# --------------------------
def run_sft_pretrain(formatted_jsonl: str, model_name: str, out_lora_dir: str):
    print("Preparing SFT dataset from:", formatted_jsonl)
    rows = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rows.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    texts = []
    for r in rows:
        messages = r["messages"]
        full = apply_chat_template_to_messages(tokenizer, messages, add_generation_prompt=False, tokenize=False)
        texts.append({"text": full})

    hf_ds = Dataset.from_list(texts)

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=CONFIG["max_seq_length"])
        out["labels"] = out["input_ids"].copy()  # causal LM labels == input_ids
        return out

    tokenized = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    import transformers
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("Loading base model via transformers:", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

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

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting SFT pre-finetune...")
    trainer.train()

    makedirs(CONFIG["checkpoints_dir"])
    out_path = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir)
    print("Saving LoRA to", out_path)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print("Saved preformat LoRA to", out_path)
    return out_path


# --------------------------
# GRPO training (train split)
# --------------------------
def run_grpo_training(formatted_jsonl: str, model_name: str, preformat_lora_path: str = None, out_lora_dir: str = None):
    if GRPOTrainer is None or SamplingParams is None:
        raise RuntimeError("trl.GRPOTrainer or vllm.SamplingParams not available. Install required libs.")

    print("Loading formatted dataset for GRPO...")
    rows = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))

    prompts = []
    answers = []
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    for r in rows:
        messages = r["messages"]
        text = apply_chat_template_to_messages(tokenizer, messages, add_generation_prompt=True, tokenize=False)
        prompts.append(text)
        raw_ans = r["raw"].get("answer")
        answers.append(raw_ans if raw_ans is not None else "")

    sampling = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=CONFIG["seed"],
        stop=[tokenizer.eos_token] if hasattr(tokenizer, "eos_token") else None,
        include_stop_str_in_output=True,
    )

    print("Loading model (FastLanguageModel) for GRPO...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=CONFIG["lora_rank"],
        gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
    )

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

    if preformat_lora_path:
        try:
            print("Loading preformat LoRA:", preformat_lora_path)
            model.load_lora(preformat_lora_path)
        except Exception as e:
            print("Could not load preformat LoRA:", e)

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
        max_prompt_length=512,
        max_completion_length=CONFIG["max_seq_length"] - 512,
        max_steps=CONFIG["grpo"]["max_steps"],
        save_steps=CONFIG["grpo"]["save_steps"],
        report_to="none",
        output_dir=CONFIG["checkpoints_dir"],
    )

    # compute 90th quantile prompt length
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

    train_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

    reward_funcs = []
    if match_format_exactly: reward_funcs.append(match_format_exactly)
    if match_format_approximately: reward_funcs.append(match_format_approximately)
    if check_answer: reward_funcs.append(check_answer)
    if check_numbers: reward_funcs.append(check_numbers)
    if not reward_funcs:
        raise RuntimeError("No reward functions available. Ensure rewards.py is present and imports succeed.")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
    )

    # Workaround for Unsloth trainer assumptions
    setattr(trainer, 'image_token_id', None)
    setattr(trainer, 'vision_start_token_id', None)
    setattr(trainer, 'vision_end_token_id', None)

    print("Starting GRPO training...")
    trainer.train()

    makedirs(CONFIG["checkpoints_dir"])
    out_dir = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir or CONFIG["grpo_lora_name"])
    model.save_lora(out_dir)
    print("Saved GRPO LoRA to", out_dir)
    return out_dir


# --------------------------
# Inference with LoRA (batched)
# --------------------------
def run_inference_with_lora(model_name: str, tokenizer_name: str, lora_path: str, prompts: List[str], max_new_tokens: int = 256, batch_size: int = None):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print("Warning: device_map/float16 shortcut failed, falling back:", e)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(CONFIG["device"])

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
    if batch_size is None:
        batch_size = CONFIG.get("eval_batch_size", 8)

    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["max_seq_length"])
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
            )

        for i in range(len(batch_prompts)):
            decoded = tokenizer.decode(gen_ids[i], skip_special_tokens=False)
            prompt_decoded = tokenizer.decode(enc["input_ids"][i], skip_special_tokens=False)
            if decoded.startswith(prompt_decoded):
                out = decoded[len(prompt_decoded):].strip()
            else:
                out = decoded.strip()
            outputs.append(out)

    return outputs


# --------------------------
# CLI orchestration — validation used for all testing
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="MedMCQA -> GRPO pipeline (validation used for testing)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="Download and process MedMCQA dataset (writes per-split files)")
    sub.add_parser("prep", help="Build chat template and format dataset for SFT/GRPO (train + validation)")
    sub.add_parser("baseline", help="Run baseline evaluation on validation split (batched)")
    sub.add_parser("sft", help="Run SFT pre-finetune on train split to teach formatting")
    sub.add_parser("grpo", help="Run GRPO training on train split (then evaluate on validation)")
    sub.add_parser("eval", help="Run inference on validation using saved LoRA")
    sub.add_parser("all", help="Run entire pipeline in sequence (download->prep->baseline->sft->grpo->eval)")

    parser.add_argument("--limit", type=int, default=5000, help="limit number of examples to process (0 or negative = no limit)")
    parser.add_argument("--model_name", type=str, default=CONFIG["model_name"])
    parser.add_argument("--processed_dir", type=str, default=CONFIG["data_dir"])
    parser.add_argument("--formatted_dir", type=str, default=CONFIG["data_dir"])
    parser.add_argument("--preformat_lora", type=str, default=os.path.join(CONFIG["checkpoints_dir"], CONFIG["preformat_lora_name"]))
    parser.add_argument("--grpo_lora", type=str, default=os.path.join(CONFIG["checkpoints_dir"], CONFIG["grpo_lora_name"]))
    parser.add_argument("--batch_size", type=int, default=CONFIG["eval_batch_size"], help="batch size for eval/inference")
    args = parser.parse_args()

    makedirs(CONFIG["data_dir"], CONFIG["checkpoints_dir"], CONFIG["experiments_dir"])

    # interpret limit: 0 or negative => no limit
    limit = args.limit if args.limit and args.limit > 0 else None

    if args.cmd == "download":
        download_medmcqa(args.processed_dir)

    elif args.cmd == "prep":
        print("Loading tokenizer to build chat template...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        # Only format train & validation (ignore test)
        for split in ["train", "validation"]:
            in_path = os.path.join(args.processed_dir, f"medmcqa_processed.{split}.jsonl")
            out_path = os.path.join(args.formatted_dir, f"medmcqa_formatted.{split}.jsonl")
            if os.path.exists(in_path):
                format_medmcqa_for_chat(in_path, out_path, tokenizer, markers, subset_limit=limit)
            else:
                print(f"Skipping split {split} (not found): {in_path}")

    elif args.cmd == "baseline":
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if not os.path.exists(val_formatted):
            print("Validation formatted file not found:", val_formatted)
            return
        baseline_evaluate_model(args.model_name, val_formatted, tokenizer_name=args.model_name, limit=limit, batch_size=args.batch_size)

    elif args.cmd == "sft":
        train_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.train.jsonl")
        if not os.path.exists(train_formatted):
            print("Train formatted file not found:", train_formatted)
            return
        out = run_sft_pretrain(train_formatted, args.model_name, CONFIG["preformat_lora_name"])
        print("SFT pretrain done, saved:", out)

    elif args.cmd == "grpo":
        train_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.train.jsonl")
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if not os.path.exists(train_formatted):
            print("Train formatted file not found:", train_formatted)
            return
        out = run_grpo_training(train_formatted, args.model_name, preformat_lora_path=args.preformat_lora, out_lora_dir=CONFIG["grpo_lora_name"])
        print("GRPO done, saved:", out)

        # quick eval on validation
        if os.path.exists(val_formatted):
            print("Running quick validation-set eval with GRPO LoRA...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
            markers = build_chat_template(tokenizer)
            prompts = []
            rows = []
            with open(val_formatted, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if limit and i >= min(200, limit): break
                    row = json.loads(line)
                    prompts.append(apply_chat_template_to_messages(tokenizer, row["messages"], add_generation_prompt=True, tokenize=False))
                    rows.append(row)
            outputs = run_inference_with_lora(args.model_name, args.model_name, args.grpo_lora, prompts, max_new_tokens=128, batch_size=args.batch_size)
            correct = 0
            total = 0
            for i, out in enumerate(outputs):
                sol = extract_pred_with_fallback(out, markers["solution_start"], markers["solution_end"], options=rows[i].get("raw", {}).get("options"))
                ex = rows[i]
                gold_label = ex.get("answer_label") or (ex.get("raw", {}) or {}).get("answer_label")
                gold_text = ex.get("answer_text") or (ex.get("raw", {}) or {}).get("answer_text")
                match = False
                if gold_label:
                    if sol and sol.strip().upper() == str(gold_label).strip().upper(): match = True
                    elif sol and str(gold_label).strip().upper() in sol.strip().upper(): match = True
                elif gold_text:
                    if sol and sol.strip().lower() in str(gold_text).strip().lower(): match = True
                print(f"[{i+1}] gold_label={gold_label} gold_text={gold_text} | pred={sol} | match={match}")
                if match: correct += 1
                total += 1
            print(f"GRPO validation eval: {correct}/{total} = {correct/total if total>0 else 0:.4f}")

    elif args.cmd == "eval":
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if not os.path.exists(val_formatted):
            print("Validation formatted file not found:", val_formatted)
            return
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        prompts = []
        rows = []
        with open(val_formatted, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= min(20, limit): break
                row = json.loads(line)
                prompts.append(apply_chat_template_to_messages(tokenizer, row["messages"], add_generation_prompt=True, tokenize=False))
                rows.append(row)
        outputs = run_inference_with_lora(args.model_name, args.model_name, args.grpo_lora, prompts, max_new_tokens=256, batch_size=args.batch_size)
        for i, out in enumerate(outputs):
            sol = extract_pred_with_fallback(out, markers["solution_start"], markers["solution_end"], options=rows[i].get("raw", {}).get("options"))
            print(f"=== Example {i} ===\nGen:\n{out}\nExtracted solution: {sol}\n")

    elif args.cmd == "all":
        # Full pipeline: download -> prep (train + validation) -> baseline (small) -> sft -> grpo -> eval (validation)
        download_medmcqa(args.processed_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        for split in ["train", "validation"]:
            in_path = os.path.join(args.processed_dir, f"medmcqa_processed.{split}.jsonl")
            out_path = os.path.join(args.formatted_dir, f"medmcqa_formatted.{split}.jsonl")
            if os.path.exists(in_path):
                # default: use full split unless --limit provided
                format_medmcqa_for_chat(in_path, out_path, tokenizer, markers, subset_limit=limit)
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if os.path.exists(val_formatted):
            baseline_evaluate_model(args.model_name, val_formatted, tokenizer_name=args.model_name, limit=50, batch_size=args.batch_size)
        train_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.train.jsonl")
        if os.path.exists(train_formatted):
            run_sft_pretrain(train_formatted, args.model_name, CONFIG["preformat_lora_name"])
            run_grpo_training(train_formatted, args.model_name, preformat_lora_path=os.path.join(CONFIG["checkpoints_dir"], CONFIG["preformat_lora_name"]))
        print("Pipeline complete. Validation split used for all testing/evaluation.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

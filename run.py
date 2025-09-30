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
def format_medmcqa_for_chat(in_jsonl: str, out_jsonl: str, tokenizer, markers: Dict[str, str], subset_limit: int = None):
    """
    Convert processed medmcqa jsonl into chat-format strings for SFT/GRPO.
    New: if the dataset contains a gold explanation under raw['exp'] (or raw['explanation']),
    place that text between the reasoning markers so SFT learns to imitate it.
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

            # pick gold label/text if available
            answer_label = ex.get("answer_label")
            answer_text = ex.get("answer_text")

            # dataset-provided explanation (preferred)
            gold_exp = None
            raw = ex.get("raw", {}) if isinstance(ex.get("raw"), dict) else {}
            # common keys: 'exp', 'explanation', 'explain'
            for k in ("exp", "explanation", "explain"):
                if isinstance(raw, dict) and raw.get(k):
                    gold_exp = str(raw.get(k)).strip()
                    break

            # construct assistant target: reasoning region contains gold_exp (if present);
            # solution region contains the gold answer label (if known) or answer_text otherwise.
            if answer_label is not None:
                sol_text = answer_label
            elif answer_text is not None and options:
                # try to derive label from answer_text
                label = None
                for j, opt in enumerate(options):
                    if answer_text.strip().lower() == opt.strip().lower() or answer_text.strip() in opt:
                        label = chr(ord("A")+j)
                        break
                if label is not None:
                    sol_text = label
                else:
                    sol_text = answer_text
            else:
                sol_text = ""

            # If gold_exp exists, place it in reasoning markers; else keep reasoning empty so SFT learns to generate.
            reasoning_content = gold_exp or ""
            assistant_content = f"{reasoning_start}{reasoning_content}{reasoning_end}{solution_start}{sol_text}{solution_end}"

            # add assistant target only if we have some gold info (so SFT can learn)
            # When the dataset has no gold label/text, we still include the prompt (for generation/GRPO).
            if sol_text or reasoning_content:
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
def run_grpo_training(formatted_jsonl: str, model_name: str, preformat_lora_path: str = None, out_lora_dir: str = None, require_label: bool = True):
    """
    Robust GRPO training entrypoint for MedMCQA.

    - formatted_jsonl: path to formatted dataset (JSONL)
    - model_name: base model HF id (for FastLanguageModel.from_pretrained)
    - preformat_lora_path: optional LoRA adapter to load before GRPO
    - out_lora_dir: output subdirectory name under CONFIG['checkpoints_dir'] to save GRPO LoRA
    - require_label: if True, filter dataset to examples that have gold label/text (cop != -1 or answer present)
    """
    if GRPOTrainer is None or SamplingParams is None:
        raise RuntimeError("trl.GRPOTrainer or vllm.SamplingParams not available. Install required libs.")

    print("Loading formatted dataset for GRPO:", formatted_jsonl)
    rows_all = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            rows_all.append(json.loads(line))
    if len(rows_all) == 0:
        raise RuntimeError(f"No examples found in {formatted_jsonl}")
    print(f"Total examples in file: {len(rows_all)}")

    # Helper: robust gold resolution (top-level or inside raw; handles cop indexing)
    def resolve_gold_from_row(row):
        gold_label = None
        gold_text = None
        if not isinstance(row, dict):
            return None, None
        if row.get("answer_label"):
            gold_label = row.get("answer_label")
        if row.get("answer_text"):
            gold_text = row.get("answer_text")
        raw = row.get("raw") or {}
        if isinstance(raw, dict):
            if gold_label is None and raw.get("answer_label"):
                gold_label = raw.get("answer_label")
            if gold_text is None and raw.get("answer_text"):
                gold_text = raw.get("answer_text")
            if gold_text is None and raw.get("answer"):
                gold_text = raw.get("answer")
            if raw.get("cop") is not None and (gold_label is None and gold_text is None):
                try:
                    idx = int(raw.get("cop"))
                    opts = raw.get("options") or row.get("options") or []
                    if idx == -1:
                        # explicit unlabeled
                        pass
                    elif 0 <= idx < len(opts):
                        gold_label = chr(ord("A") + idx)
                        gold_text = opts[idx]
                    elif 1 <= idx <= len(opts):
                        gold_label = chr(ord("A") + (idx - 1))
                        gold_text = opts[idx - 1]
                except Exception:
                    s = str(raw.get("cop")).strip()
                    if s:
                        gold_text = gold_text or s
        return gold_label, gold_text

    # Filter to labeled examples if requested
    rows = rows_all
    if require_label:
        kept = []
        for r in rows_all:
            lbl, txt = resolve_gold_from_row(r)
            if lbl is not None or txt is not None:
                kept.append(r)
        print(f"Filtered training rows: kept {len(kept)}/{len(rows_all)} examples with gold label/text.")
        rows = kept
        if len(rows) == 0:
            raise RuntimeError("No labeled examples after filtering. Set require_label=False or fix processed dataset (cop != -1 or answer_label present).")

    # Tokenizer + markers
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    # Build prompts + answers arrays
    prompts = []
    answers = []
    for r in rows:
        prompts.append(apply_chat_template_to_messages(tokenizer, r["messages"], add_generation_prompt=True, tokenize=False))
        lbl, txt = resolve_gold_from_row(r)
        answers.append(txt or lbl or "")

    print(f"Prepared {len(prompts)} prompts for GRPO (after filtering).")

    # vllm sampling params
    sampling = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=CONFIG.get("seed", 3407),
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

    # Apply PEFT/LoRA
    print("Applying PEFT/LoRA wrappers to model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=CONFIG["lora_rank"] * 2,
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG.get("seed", 3407),
    )

    if preformat_lora_path:
        try:
            print("Loading preformat LoRA:", preformat_lora_path)
            model.load_lora(preformat_lora_path)
            print("Preformat LoRA loaded.")
        except Exception as e:
            print("Could not load preformat LoRA (continuing):", e)

    # GRPO args
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

    # Compute prompt length quantile
    print("Tokenizing prompts to compute prompt length quantile...")
    token_lens = []
    for p in prompts:
        try:
            ids = tokenizer(p, return_tensors="pt")["input_ids"][0]
            token_lens.append(int(ids.shape[0]))
        except Exception:
            token_lens.append(min(len(p.split()), CONFIG["max_seq_length"] // 4))
    import numpy as np
    try:
        max_prompt_length = int(np.quantile(np.array(token_lens), 0.9))
    except Exception:
        max_prompt_length = min(max(token_lens) if token_lens else 512, 512)
    grpo_args.max_prompt_length = max_prompt_length + 1
    grpo_args.max_completion_length = CONFIG["max_seq_length"] - grpo_args.max_prompt_length
    print(f"Set max_prompt_length={grpo_args.max_prompt_length}, max_completion_length={grpo_args.max_completion_length}")

    # Build HF dataset
    train_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

    # Collect reward functions
    raw_reward_fns = []
    if match_format_exactly: raw_reward_fns.append(match_format_exactly)
    if match_format_approximately: raw_reward_fns.append(match_format_approximately)
    if check_answer: raw_reward_fns.append(check_answer)
    if check_numbers: raw_reward_fns.append(check_numbers)
    if not raw_reward_fns:
        raise RuntimeError("No reward functions available. Ensure rewards.py is present and imports succeed.")

    print(f"Wrapping {len(raw_reward_fns)} reward functions for GRPO trainer (batch-aware).")

    # Debug control
    first_call_debug = {"seen": False, "printed_sample": False}

    # Batch-aware wrapper
    def make_batch_wrapper(fn):
        name = getattr(fn, "__name__", "reward_fn")

        def wrapped(completions, prompts_arg=None, completion_ids=None, **kwargs):
            # One-time debug print
            if not first_call_debug["seen"]:
                print("=== REWARD WRAPPER FIRST CALL DEBUG ===")
                print("wrapped fn:", name)
                print("completions type:", type(completions))
                if isinstance(completions, (list, tuple)):
                    print("len(completions):", len(completions))
                    if len(completions) > 0:
                        preview = completions[0]
                        print("first completion preview (truncated):", (str(preview)[:500] + "...") if preview else None)
                print("prompts_arg type:", type(prompts_arg))
                print("completion_ids type:", type(completion_ids), "sample:", (completion_ids[:4] if isinstance(completion_ids, (list,tuple)) else completion_ids))
                first_call_debug["seen"] = True

            flat_comps = []
            comp_to_idx = []

            # If completion_ids provided and elements are scalar ints -> try to use them.
            # If completion_ids elements are lists (token ids), treat them as token lists and fallback to positional mapping.
            if isinstance(completion_ids, (list, tuple)):
                # If completions align with completion_ids by length
                if isinstance(completions, (list, tuple)) and len(completions) == len(completion_ids):
                    for i, comp_entry in enumerate(completions):
                        target_idx = completion_ids[i]
                        # If trainer provided token-id list here, interpret as "not an index" -> fallback to i
                        if isinstance(target_idx, (list, tuple)):
                            mapped_idx = i
                        else:
                            # try to coerce to int when possible
                            try:
                                mapped_idx = int(target_idx)
                            except Exception:
                                mapped_idx = i
                        if isinstance(comp_entry, (list, tuple)):
                            for cc in comp_entry:
                                flat_comps.append(cc)
                                comp_to_idx.append(mapped_idx)
                        else:
                            flat_comps.append(comp_entry)
                            comp_to_idx.append(mapped_idx)
                else:
                    # fallback flatten & map by index where possible
                    if isinstance(completions, (list, tuple)):
                        for i, comp_entry in enumerate(completions):
                            mapped_idx = None
                            if i < len(completion_ids):
                                cand = completion_ids[i]
                                if isinstance(cand, (list, tuple)):
                                    mapped_idx = i
                                else:
                                    try:
                                        mapped_idx = int(cand)
                                    except Exception:
                                        mapped_idx = i
                            else:
                                mapped_idx = i
                            if isinstance(comp_entry, (list, tuple)):
                                for cc in comp_entry:
                                    flat_comps.append(cc)
                                    comp_to_idx.append(mapped_idx)
                            else:
                                flat_comps.append(comp_entry)
                                comp_to_idx.append(mapped_idx)
                    else:
                        flat_comps = [completions]
                        comp_to_idx = [0]
            else:
                # No completion_ids: rely on structure of completions and prompts_arg
                if isinstance(completions, (list, tuple)):
                    # If nested lists and prompts_arg is list -> map sublists to prompt indices
                    if len(completions) > 0 and isinstance(completions[0], (list, tuple)) and isinstance(prompts_arg, (list, tuple)):
                        for pi, sub in enumerate(completions):
                            for cc in (sub if isinstance(sub, (list, tuple)) else [sub]):
                                flat_comps.append(cc)
                                comp_to_idx.append(pi if pi < len(prompts_arg) else 0)
                    else:
                        for i, c in enumerate(completions):
                            flat_comps.append(c)
                            comp_to_idx.append(i if isinstance(prompts_arg, (list, tuple)) and i < len(prompts_arg) else 0)
                else:
                    flat_comps = [completions]
                    comp_to_idx = [0]

            # Compute reward for each flattened completion
            rewards_out = []
            for comp, ex_idx in zip(flat_comps, comp_to_idx):
                ex_obj = rows[ex_idx] if (isinstance(ex_idx, int) and 0 <= ex_idx < len(rows)) else None
                prompt_for_call = None
                if isinstance(prompts_arg, (list, tuple)) and isinstance(ex_idx, int) and ex_idx < len(prompts_arg):
                    prompt_for_call = prompts_arg[ex_idx]
                elif isinstance(prompts_arg, str):
                    prompt_for_call = prompts_arg

                val = None
                try:
                    if ex_obj is not None:
                        try:
                            val = fn(comp, prompt_for_call, ex_obj, markers)
                        except TypeError:
                            try:
                                val = fn(comp, prompt_for_call, ex_obj)
                            except TypeError:
                                val = None
                    if val is None:
                        try:
                            val = fn(comp, prompt_for_call)
                        except TypeError:
                            try:
                                val = fn(comp)
                            except Exception:
                                val = None
                except Exception as e:
                    print(f"Reward function {name} raised for ex_idx={ex_idx}: {e}")
                    val = None

                # normalize to float
                if isinstance(val, (int, float)):
                    rewards_out.append(float(val))
                elif isinstance(val, (list, tuple)):
                    try:
                        rewards_out.append(float(val[0]) if len(val) > 0 else 0.0)
                    except Exception:
                        rewards_out.append(0.0)
                else:
                    rewards_out.append(0.0)

            if first_call_debug.get("seen") and not first_call_debug.get("printed_sample"):
                print("=== REWARD WRAPPER SAMPLE OUTPUT ===")
                print("flat_comps_count:", len(flat_comps))
                print("comp_to_idx sample:", comp_to_idx[:10])
                print("rewards_out sample:", rewards_out[:10])
                first_call_debug["printed_sample"] = True

            return rewards_out

        return wrapped

    # Wrap raw reward functions
    wrapped_reward_funcs = [make_batch_wrapper(fn) for fn in raw_reward_fns]

    # Create GRPOTrainer
    print("Creating GRPOTrainer with args:", grpo_args)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=wrapped_reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
    )

    # Workaround: clear vision tokens for text-only models
    setattr(trainer, 'image_token_id', None)
    setattr(trainer, 'vision_start_token_id', None)
    setattr(trainer, 'vision_end_token_id', None)

    print("Starting GRPO training (this may take a while)...")
    trainer.train()

    # Save LoRA adapter
    makedirs(CONFIG["checkpoints_dir"])
    out_dir = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir or CONFIG["grpo_lora_name"])
    try:
        model.save_lora(out_dir)
        print("Saved GRPO LoRA to", out_dir)
    except Exception as e:
        print("Failed to save GRPO LoRA via model.save_lora():", e)
        try:
            trainer.save_model(out_dir)
            print("Trainer saved model to", out_dir)
        except Exception as e2:
            print("Also failed to save via trainer.save_model():", e2)
            raise

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

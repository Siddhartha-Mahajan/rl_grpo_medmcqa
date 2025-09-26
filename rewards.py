# rewards.py
"""
Reward functions for GRPO (used by run.py).
Implements:
  - match_format_exactly(completions, **kwargs)
  - match_format_approximately(completions, **kwargs)
  - check_answer(prompts, completions, answer, **kwargs)
  - check_numbers(prompts, completions, answer, **kwargs)

Expectations about `completions`:
  - Each element in `completions` is a list-like where the first item is a dict containing "content":
      e.g. completion = [ {"content": "<start_working_out>...<end_working_out><SOLUTION>42</SOLUTION>"} ]
  - This matches the pattern used in the run.py you were given.
"""

import re
from typing import List, Any, Optional

# Markers — must match the markers used when formatting prompts in run.py
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

# Regex to match the reasoning_end ... <SOLUTION> answer </SOLUTION>
# This finds the text inside the SOLUTION markers (group 1) robustly, allowing optional trailing eos/token
_solution_end_regex = re.escape(SOLUTION_END) + r"[\s]{0,}" + r"(?:<\|endoftext\|>)?"  # allow a generic eos token if present
_match_format_pattern = re.compile(
    rf"{re.escape(REASONING_END)}.*?"\
    rf"{re.escape(SOLUTION_START)}(.+?){_solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

# Regex to extract a number (first numeric group) within the SOLUTION block.
# Allows commas and decimals and optional leading minus.
_match_numbers = re.compile(
    re.escape(SOLUTION_START) + r".*?[-]?[\d\.\,]{1,}" ,
    flags=re.MULTILINE | re.DOTALL
)
# More specific capture of the numeric substring inside SOLUTION tags:
_match_numbers_capture = re.compile(
    re.escape(SOLUTION_START) + r".*?([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL
)

def _get_completion_text(completion_item: Any) -> Optional[str]:
    """
    Normalize a single `completion` entry to string.
    We expect a list-like where completion_item[0]["content"] is the text,
    but be defensive and accept dicts or strings.
    """
    if completion_item is None:
        return None
    try:
        # completion_item might be list-like: [ {"content": "..."} ]
        if isinstance(completion_item, (list, tuple)) and len(completion_item) > 0:
            first = completion_item[0]
            if isinstance(first, dict) and "content" in first:
                return first["content"]
            # maybe nested one more level
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict) and "content" in first[0]:
                return first[0]["content"]
        # If dict with content directly
        if isinstance(completion_item, dict) and "content" in completion_item:
            return completion_item["content"]
        # If string
        if isinstance(completion_item, str):
            return completion_item
    except Exception:
        return None
    return None


# ---------------------------
# Reward functions
# ---------------------------

def match_format_exactly(completions: List[Any], **kwargs) -> List[float]:
    """
    Reward = +3.0 when the generated response contains reasoning_end followed by <SOLUTION>...</SOLUTION>
    Otherwise 0.
    Signature matches what run.py expects: (completions, **kwargs)
    """
    scores = []
    for c in completions:
        text = _get_completion_text(c)
        score = 0.0
        if text is not None:
            if _match_format_pattern.search(text) is not None:
                score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions: List[Any], **kwargs) -> List[float]:
    """
    Soft reward for seeing reasoning_end, <SOLUTION> and </SOLUTION> tokens.
    +0.5 for each correct presence; -1.0 penalty if missing.
    """
    scores = []
    for c in completions:
        text = _get_completion_text(c) or ""
        score = 0.0
        # reasoning_end is often prepended externally, so we skip rewarding it here.
        score += 0.5 if text.count(REASONING_END) == 1 else -1.0
        score += 0.5 if text.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if text.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts: List[Any], completions: List[Any], answer: List[Any], **kwargs) -> List[float]:
    """
    Extract the content inside <SOLUTION>...</SOLUTION> using regex.
    Reward rules:
      - if exact string match -> +5.0
      - if stripped match -> +3.5
      - if both numeric and ratio within 0.9-1.1 -> +2.0
      - if numeric and ratio within 0.8-1.2 -> +1.5
      - otherwise penalize -2.5 or -4.5 for parse errors.
    Note: `answer` is expected to be a list of gold answers (strings).
    """
    scores = []
    for idx, (c, gold) in enumerate(zip(completions, answer)):
        text = _get_completion_text(c) or ""
        score = 0.0
        match = _match_format_pattern.search(text)
        if match is None:
            # no solution block found -> penalize
            scores.append(-2.0)
            continue
        guess = match.group(1).strip()
        # exact match
        if gold is None:
            scores.append(-1.5)
            continue
        gold_str = str(gold).strip()
        if guess == gold_str:
            score += 5.0
            scores.append(score)
            continue
        if guess.strip() == gold_str.strip():
            score += 3.5
            scores.append(score)
            continue
        # if numeric, compare ratios
        try:
            g = float(guess.replace(",", ""))
            t = float(gold_str.replace(",", ""))
            ratio = g / t if t != 0 else float('inf')
            if 0.9 <= ratio <= 1.1:
                score += 2.0
            elif 0.8 <= ratio <= 1.2:
                score += 1.5
            else:
                score -= 2.5
        except Exception:
            score -= 4.5
        scores.append(score)
    return scores


# Print occasionally for debugging (the run.py uses a similar pattern)
_PRINT_COUNTER = 0
_PRINT_EVERY = 5


def check_numbers(prompts: List[Any], completions: List[Any], answer: List[Any], **kwargs) -> List[float]:
    """
    Extract numeric answers inside <SOLUTION> and compare to the numerical gold answer.
    Returns +3.5 for exact float equality, -1.5 for mismatches, -2.5 if no extraction.
    Also prints one sample every PRINT_EVERY steps to help debugging (as in the original snippet).
    """
    global _PRINT_COUNTER, _PRINT_EVERY
    scores = []
    for c, gold in zip(completions, answer):
        text = _get_completion_text(c) or ""
        # Extract numeric substring from the SOLUTION block
        num_match = _match_numbers_capture.search(text)
        if num_match is None:
            scores.append(-2.5)
            continue
        guess_str = num_match.group(1).strip()
        try:
            guess_val = float(guess_str.replace(",", ""))
            true_val = float(str(gold).strip().replace(",", ""))
            if guess_val == true_val:
                scores.append(3.5)
            else:
                scores.append(-1.5)
        except Exception:
            scores.append(0.0)
            continue

        # occasional print for debugging (useful in long runs)
        if _PRINT_COUNTER % _PRINT_EVERY == 0:
            try:
                # small, safe print
                print("DEBUG check_numbers sample:", "gold=", gold, "extracted=", guess_str)
            except Exception:
                pass
        _PRINT_COUNTER += 1
    return scores


# ---------------------------
# Unit tests / quick checks
# ---------------------------
if __name__ == "__main__":
    # Quick smoke tests to verify behavior
    sample_good = [
        [ {"content": f"blah {REASONING_END} {SOLUTION_START} 42 {SOLUTION_END}"} ]
    ]
    sample_bad = [
        [ {"content": "no solution here"} ]
    ]
    print("match_format_exactly (good):", match_format_exactly(sample_good))
    print("match_format_exactly (bad):", match_format_exactly(sample_bad))
    print("match_format_approximately (good):", match_format_approximately(sample_good))
    print("match_format_approximately (bad):", match_format_approximately(sample_bad))

    # check_answer
    prompts = [[]]  # unused in this function
    answers = ["42"]
    print("check_answer (good):", check_answer(prompts, sample_good, answers))
    print("check_answer (bad):", check_answer(prompts, sample_bad, answers))

    # check_numbers
    print("check_numbers (good):", check_numbers(prompts, sample_good, answers))
    print("check_numbers (bad):", check_numbers(prompts, sample_bad, answers))

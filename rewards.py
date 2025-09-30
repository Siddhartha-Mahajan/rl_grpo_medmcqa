# rewards.py
import re
from typing import Optional
# optional embedding fallback
EMBEDDING_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

def extract_between(text: str, start: str, end: str) -> Optional[str]:
    if not text:
        return None
    s = text.find(start)
    if s == -1:
        return None
    s2 = text.find(end, s + len(start))
    if s2 == -1:
        return text[s + len(start):].strip()
    return text[s + len(start):s2].strip()

def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s.strip().lower())

def jaccard_similarity(a: Optional[str], b: Optional[str]) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    sa = set(a.split()); sb = set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def embedding_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not EMBEDDING_AVAILABLE:
        return jaccard_similarity(a, b)
    if not a or not b:
        return 0.0
    emb_a = EMBEDDING_MODEL.encode(a, convert_to_tensor=True)
    emb_b = EMBEDDING_MODEL.encode(b, convert_to_tensor=True)
    cos = util.cos_sim(emb_a, emb_b).item()
    # normalize from [-1,1] to [0,1]
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))

def _derive_gold_label_and_text(example: dict):
    gold_label = example.get("answer_label")
    gold_text = example.get("answer_text")
    raw = example.get("raw") or {}
    if gold_label is None:
        # try raw fields
        if isinstance(raw, dict):
            if raw.get("answer_label"):
                gold_label = raw.get("answer_label")
            # cop numeric index
            cop = raw.get("cop")
            if cop is not None:
                try:
                    idx = int(cop)
                    opts = raw.get("options") or example.get("options") or []
                    if idx == -1:
                        gold_label = None
                    elif 0 <= idx < len(opts):
                        gold_label = chr(ord("A") + idx)
                        gold_text = opts[idx]
                    elif 1 <= idx <= len(opts):
                        gold_label = chr(ord("A") + (idx - 1))
                        gold_text = opts[idx-1]
                except Exception:
                    # cop not int -> maybe text
                    s = str(cop).strip()
                    if s:
                        gold_text = gold_text or s
    if gold_text is None and isinstance(raw, dict) and raw.get("answer"):
        gold_text = raw.get("answer")
    return gold_label, gold_text

def match_format_exactly(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    """
    1.0 if the extracted solution equals gold label, else 0.0
    """
    gold_label, _ = _derive_gold_label_and_text(example)
    if not gold_label:
        return 0.0
    # try explicit solution extraction
    pred = extract_between(gen_text or "", markers.get("solution_start"), markers.get("solution_end"))
    if not pred:
        # fallback single letter search
        m = re.search(r'\b([A-Z])\b', (gen_text or "").upper())
        if m: pred = m.group(1)
    if pred and pred.strip().upper() == str(gold_label).strip().upper():
        return 1.0
    # also accept if gold label occurs anywhere uppercase
    if pred and str(gold_label).strip().upper() in pred.strip().upper():
        return 1.0
    return 0.0

def match_explanation_similarity(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    """
    Soft similarity (0..1) between generated reasoning and dataset 'exp' (if present).
    Uses embeddings if available, else Jaccard token overlap.
    """
    raw = example.get("raw") or {}
    gold_exp = None
    for k in ("exp", "explanation", "explain"):
        if isinstance(raw, dict) and raw.get(k):
            gold_exp = str(raw.get(k)).strip()
            break
    if not gold_exp:
        return 0.0
    gen_reasoning = extract_between(gen_text or "", markers.get("reasoning_start"), markers.get("reasoning_end"))
    if not gen_reasoning:
        # fallback: take text before solution tag
        solidx = (gen_text or "").find(markers.get("solution_start") or "")
        if solidx > 0:
            gen_reasoning = (gen_text or "")[:solidx]
        else:
            return 0.0
    # compute similarity
    return embedding_similarity(gen_reasoning, gold_exp)

def check_answer(gen_text: str, prompt: str, example: dict, markers: dict, label_weight: float = 1.0, exp_weight: float = 0.4) -> float:
    lbl = match_format_exactly(gen_text, prompt, example, markers)
    exp_sim = match_explanation_similarity(gen_text, prompt, example, markers)
    return float(lbl * label_weight + exp_sim * exp_weight)

# alias/compatibility
def match_format_approximately(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    return check_answer(gen_text, prompt, example, markers, label_weight=0.9, exp_weight=0.3)

def check_numbers(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    # not used for MCQ, return 0
    return 0.0

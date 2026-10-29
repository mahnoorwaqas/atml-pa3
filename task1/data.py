"""MetaMathQA SFT data: eligibility filtering, chat formatting, response-only label masking."""
import re, os
import torch

from common.utils import save_json

SEED = 6304


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prompt_ids(tok, question):
    text = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_example(tok, question, response):
    """Returns (input_ids, labels, formatted_len). Loss only on response tokens + end-of-turn token."""
    p = prompt_ids(tok, question)
    r = tok(response, add_special_tokens=False)["input_ids"]
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    ids = p + r + [eot]
    labels = [-100] * len(p) + r + [eot]
    nl = len(tok("\n", add_special_tokens=False)["input_ids"])            # template's trailing newline
    return ids, labels, len(ids) + nl


def build_training_set(tok, rows, gsm_test_questions, n=40000, max_len=2048):
    """rows: iterable of dicts (query, response, original_question) in SOURCE ORDER.
    Takes the first n eligible examples; no shuffling, no truncation."""
    test_norm = {normalize_text(q) for q in gsm_test_questions}
    out, ids_meta, skipped = [], [], {"empty": 0, "too_long": 0, "gsm8k_test": 0}
    for i, r in enumerate(rows):
        q, a = (r.get("query") or "").strip(), (r.get("response") or "").strip()
        if not q or not a:
            skipped["empty"] += 1; continue
        if normalize_text(q) in test_norm or normalize_text(r.get("original_question") or "") in test_norm:
            skipped["gsm8k_test"] += 1; continue
        ids, labels, flen = encode_example(tok, q, a)
        if flen > max_len:
            skipped["too_long"] += 1; continue
        out.append({"input_ids": ids, "labels": labels}); ids_meta.append(i)
        if len(out) >= n:
            break
    return out, {"source_indices": ids_meta, "skipped": skipped, "n": len(out)}


def load_metamath_rows():
    from datasets import load_dataset
    return load_dataset("meta-math/MetaMathQA", split="train", streaming=True)     # source order


def load_gsm8k_test():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split="test")


def prepare(tok, out_path, n=40000, rows=None, gsm_questions=None):
    rows = rows if rows is not None else load_metamath_rows()
    gsm_questions = gsm_questions if gsm_questions is not None else [x["question"] for x in load_gsm8k_test()]
    data, meta = build_training_set(tok, rows, gsm_questions, n)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(data, out_path)
    save_json(meta, out_path + ".meta.json")
    return data, meta


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"]); lab[i, :L] = torch.tensor(b["labels"]); att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="results/task1/train_data.pt"); ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()
    prepare(AutoTokenizer.from_pretrained(a.model), a.out, a.n)

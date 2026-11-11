"""Write the 50 live-generation questions (first 50 held-out MATH questions of the seeded permutation).
Usage: python -m task3.make_q50 --model_dir <LLM folder in ATML-PA3-Data> --out q50.jsonl [--schema schema.json]
The split is recomputed from the supplied question ids with the same function / seed as task3.run_offline,
so this can run before (or in parallel with) the offline pipeline.
"""
import argparse, json, os
import pyarrow.parquet as pq
from common.utils import SEED
from .data import make_splits
from .schema import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--schema")
    a = ap.parse_args()
    schema = json.load(open(a.schema)) if a.schema else None
    path = os.path.join(a.model_dir, "generations.parquet")
    names = pq.ParquetFile(path).schema_arrow.names
    qc = resolve(names, "question", schema, required=False)
    gc = resolve(names, "gold", schema)
    if qc is None:
        raise SystemExit(f"No question-text column in generations.parquet ({names}). Pass --schema with a 'question' mapping, "
                         "or take the problem text from the dataset named in README.MD.")
    df = pq.read_table(path, columns=["sample_id", qc, gc]).to_pandas().drop_duplicates("sample_id")
    df["sample_id"] = df["sample_id"].astype(str)
    ids = sorted(df["sample_id"])
    sp = make_splits(ids, SEED, 200 if len(ids) >= 400 else max(1, len(ids) // 5))
    live = sp["live50"]
    by = df.set_index("sample_id")
    with open(a.out, "w") as f:
        for q in live:
            f.write(json.dumps({"qid": q, "problem": str(by.loc[q, qc]), "gold": str(by.loc[q, gc])}) + "\n")
    print(f"wrote {len(live)} questions to {a.out} (of {len(ids)} total)")


if __name__ == "__main__":
    main()

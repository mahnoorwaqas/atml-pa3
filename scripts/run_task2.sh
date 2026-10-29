#!/usr/bin/env bash
# Task 2 end-to-end (GPU).  COCO=<coco2014 root>  VQA=<VQAv2 root with complementary pairs>
set -e
python -m task2.data --coco_root "$COCO" --vqa_root "$VQA" --out results/task2/dataset.json
python - <<PY
import json, torch
from task2.features import cache_features
ds = json.load(open("results/task2/dataset.json"))
print(cache_features(ds["image_paths"], "results/task2/feats"))
PY
python -m task2.train --dataset results/task2/dataset.json --features results/task2/feats --out results/task2/main
python -m task2.evaluate --dataset results/task2/dataset.json --features results/task2/feats --run results/task2/main --out results/task2/main/final
python -m task2.analyze --run results/task2/main
# optional: 49-token compression run
# python -m task2.train --dataset ... --features ... --out results/task2/tok49 --set n_tokens=49

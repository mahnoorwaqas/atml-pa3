#!/usr/bin/env bash
# Task 3 (CPU for offline parts).  LLM_DIR / VLM_DIR are the model folders inside ATML-PA3-Data.
set -e
python -m task3.run_offline --kind math --model_dir "$LLM_DIR" --out results/task3/math
python -m task3.run_offline --kind mathvista --model_dir "$VLM_DIR" --meta mathvista_testmini_meta.json --out results/task3/mathvista
python -m task3.analyze --dir results/task3/math; python -m task3.analyze --dir results/task3/mathvista
# live generation (GPU): write the 50 ids from results/task3/math/splits.json['live50'] to q50.jsonl first
python -m task3.live --model Qwen/Qwen2.5-3B-Instruct --revision "$QWEN_REV" --questions q50.jsonl --out results/task3/live

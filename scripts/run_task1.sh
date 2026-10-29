#!/usr/bin/env bash
# Task 1 end-to-end (GPU).  Needs first_plot_questions.yaml in the repo root.
set -e
python -m task1.data --out results/task1/train_data.pt
for m in full l2sp lora4 lora16 oplora; do python -m task1.train --method $m --out results/task1/$m; done
python -m task1.evaluate --runs_dir results/task1 --out results/task1/eval
python -m task1.diagnostics --runs_dir results/task1
python -m task1.qualitative --runs_dir results/task1 --questions first_plot_questions.yaml
python -m task1.analyze

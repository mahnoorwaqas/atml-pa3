# ATML PA3 — LLM fine-tuning, VLM grounding, test-time compute (seed 6304)

```
common/   utils, LoRA + OPLoRA (from scratch), tiny offline test assets
task1/    data · train (Full/L2-SP/LoRA-4/LoRA-16/OPLoRA) · evaluate (merge, lm-eval) · diagnostics · qualitative · analyze
task2/    data · features (SigLIP cache) · model (connector+Qwen) · train (3 stages) · evaluate (dev + 4-condition final) · analyze
task3/    schema · answers · data · engine · signals · adaptive · live · run_offline · analyze
configs/  task1.yaml, task2.yaml          scripts/  run_task{1,2,3}.sh          tests/  smoke tests
```
Setup: `pip install -r requirements.txt`. Smoke tests (CPU, no downloads): `python -m tests.run_smoke`.

## Reproducing
* **Task 1** (`scripts/run_task1.sh`): builds the 40,000-example MetaMathQA set (decontaminated against GSM8K test), trains the five
  variants (1,250 updates each; checkpoints at 312/625/937/1250; resume with `--resume results/task1/<run>/<update>`), evaluates the
  7 conditions (adapters merged into *fresh* backbones, merged-vs-unmerged logit check), diagnostics (ρ_k, η, R(z)) on down_proj layers 0/13/27,
  qualitative generation (168 responses; 42-response annotation CSV with **blank manual columns — annotate yourself**).
* **Task 2** (`scripts/run_task2.sh`): dataset builder → SigLIP feature cache → 4,500-update curriculum → final grounding eval
  (correct / swap / mismatch / no-image; 4,000-resample pair bootstrap). `--set n_tokens=49` for the optional compression run.
* **Task 3** (`scripts/run_task3.sh`): offline pipeline on the supplied rollouts, figures/case studies, live 3-strategy comparison.

## Things you must check against your materials (I could not see them)
1. **Task 3 column names.** The supplied README.MD was not available. `task3/schema.py` maps logical fields → columns by candidate names;
   if yours differ, pass `--schema schema.json` (`{"pred": "my_col", ...}`). Misaligned token lists raise/report via `verify_alignment`.
2. **MathVista metadata** (`question_type, answer_type, choices, precision`, keyed by `pid` = `sample_id`) is read from `--meta`.
3. **Live prompt**: `PROBLEM_SEP` in `task3/live.py` (problem + separator + instruction) and the Qwen2.5 `--revision` must match README.MD.
4. **Live-50 subset** = positions 200–249 of the seeded permutation (`splits.json["live50"]`), i.e. the first 50 held-out questions.
5. `first_plot_questions.yaml` is parsed as `id / paraphrases / system` with `_json` / `_template` suffixes; asserts 24 prompts.
6. lm-eval: `gsm8k_cot` run zero-shot with chat template + `enable_thinking=False` (injected by wrapping `apply_chat_template`); other
   tasks without chat template. Check task names against your installed lm-eval version.

## Design notes / interpretations
* Warm-up 38 = round-half-up(0.03·1250); cosine decays to 0. Checkpoint updates = floor(frac·1250) → 312/625/937/1250.
* SFT loss is normalised by valid response tokens of the effective batch; L2-SP penalty (λ=1e-3) is added once per update over all
  unique parameters (tied weights once) via the grad scaler; SFT loss and total objective are logged separately.
* OPLoRA uses effective factors `B~ = B − U_k(U_kᵀB)`, `A~ = A − (A V_k)V_kᵀ`; bases computed once in FP32; projected factors are exported and merged.
* SigLIP: HF `last_hidden_state` already includes the final layernorm, so it is applied exactly once. Connector output is multiplied by a fixed
  scalar (median text-embedding norm / median connector-output norm on a training batch; saved in `calibration.json`).
* Task 2 VQA normalisation is a compact re-implementation of the official one (short contraction list).
* Task 3: groups come from candidates only (math-verify, clustering by cheap key then pairwise `verify`); MathVista via type/choice/precision rules.
  `math_verify` timeouts are disabled (signal-based timeouts break in worker threads) — a pathological string could stall; interrupt and inspect if so.
  VisConf calibration, ranks and scores are recomputed on each candidate *prefix* only; adaptive rules never read gold answers or unrevealed rollouts.
* Report text must be written by you (assignment AI policy); code here is LLM-assisted — you must understand every line and cite reused code.

## What was verified (CPU smoke tests with tiny random Qwen3/SigLIP and synthetic data)
All five Task 1 methods train, log, checkpoint, resume bit-identically (to 1e-8) and merge (logit gap ~3e-7); OPLoRA's ρ_k = 0 exactly and its
projection constraints hold numerically; Task 2 builds isolated splits, trains the 3 stages (resume mid-stage 2 reproduces), evaluates all
conditions with pair bootstrap; Task 3 runs offline (LLM + VLM), adaptive policies, signal analysis, and live generation on a tiny model.
**Not run here:** real Qwen/SigLIP weights, GPU/FP16 AMP paths, lm-evaluation-harness, real COCO/VQAv2/MATH/MathVista data (no GPU or internet).
Expect to debug small API/version/column-name issues on first real runs.

"""Back up / restore Task 1 runs on the Hugging Face Hub so Kaggle/Colab disconnects lose nothing.

Checkpoint files written by task1.train (flat layout in the run dir):
  model_u<N>.pt, train_state_u<N>.pt, adapter_u<N>.pt (adapter runs), plus train_log.jsonl, ckpt_meta.jsonl, summary.json

Train (resumes automatically from the newest complete checkpoint on the Hub):
  python -m task1.hf_sync --method full --repo pa3-fullft --out results/task1/full --prune_local --hf_keep 625 1250 [--set k=v ...]
Fetch specific checkpoints for evaluation (no training):
  python -m task1.hf_sync --pull_only --repo pa3-fullft --out results/task1/full --steps 625 1250
"""
import argparse, os, re, subprocess, sys, threading, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

SMALL = ["train_log.jsonl", "ckpt_meta.jsonl", "summary.json"]
STABLE_SECS = 60


def api_and_repo(name):
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=tok)
    rid = name if "/" in name else f"{api.whoami()['name']}/{name}"
    api.create_repo(rid, private=True, exist_ok=True)
    return api, rid


def step_files(u):
    return [f"model_u{u}.pt", f"train_state_u{u}.pt", f"adapter_u{u}.pt"]


def local_steps(run):
    return sorted(int(m.group(1)) for f in Path(run).glob("train_state_u*.pt") if (m := re.match(r"train_state_u(\d+)\.pt", f.name)))


def remote_steps(api, rid):
    files = set(api.list_repo_files(rid))
    steps = {int(m.group(1)) for f in files if (m := re.match(r"train_state_u(\d+)\.pt", f))}
    return sorted(u for u in steps if f"model_u{u}.pt" in files), files


def stable(run, u):
    fs = [Path(run) / f for f in step_files(u) if (Path(run) / f).exists()]
    return bool(fs) and all(time.time() - f.stat().st_mtime > STABLE_SECS for f in fs)


def upload_small(api, rid, run):
    for f in SMALL:
        p = Path(run) / f
        if p.exists():
            try:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=f, repo_id=rid, commit_message=f"update {f}")
            except Exception as e:
                print(f"[sync] {f} upload failed: {e}", flush=True)


def sync_once(api, rid, run, keep_hf, prune_local, force=False):
    done, files = remote_steps(api, rid)
    for u in local_steps(run):
        if u in done or not (force or stable(run, u)):
            continue
        for f in step_files(u):
            p = Path(run) / f
            if p.exists():
                print(f"[sync] uploading {f} ({p.stat().st_size / 2**30:.2f} GB)", flush=True)
                api.upload_file(path_or_fileobj=str(p), path_in_repo=f, repo_id=rid, commit_message=f"ckpt {u}: {f}")
        done.append(u)
    upload_small(api, rid, run)
    done, files = remote_steps(api, rid)
    if not done:
        return
    newest = max(done)
    if prune_local:  # everything older than the newest *uploaded* checkpoint is already safe on the Hub
        for u in local_steps(run):
            if u < newest:
                for f in step_files(u):
                    if (Path(run) / f).exists() and f in files:
                        (Path(run) / f).unlink(); print(f"[sync] removed local {f}", flush=True)
    if keep_hf is not None:  # drop remote model/state of steps that are neither newest nor in keep_hf
        for u in done:
            if u != newest and u not in keep_hf:
                for f in step_files(u):
                    if f in files and not f.startswith("adapter_"):
                        api.delete_file(f, repo_id=rid, commit_message=f"drop {f}"); print(f"[sync] removed remote {f}", flush=True)


def download(api, rid, run, names):
    Path(run).mkdir(parents=True, exist_ok=True)
    files = set(api.list_repo_files(rid))
    for f in names:
        if f in files and not (Path(run) / f).exists():
            print(f"[sync] downloading {f}", flush=True)
            hf_hub_download(rid, f, local_dir=str(run), token=os.environ["HF_TOKEN"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--final_update", type=int, default=1250)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--hf_keep", type=int, nargs="*", default=None, help="steps whose model files stay on the Hub (plus newest)")
    ap.add_argument("--prune_local", action="store_true")
    ap.add_argument("--pull_only", action="store_true"); ap.add_argument("--steps", type=int, nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=None)                       # keep last
    a = ap.parse_args()
    api, rid = api_and_repo(a.repo)
    run = Path(a.out)
    if a.pull_only:
        download(api, rid, run, SMALL + [f for u in a.steps for f in step_files(u) if not f.startswith("train_state")])
        return
    if not a.method:
        raise SystemExit("--method is required for training")
    done, _ = remote_steps(api, rid)
    n = max(done) if done else None
    if n is not None:
        download(api, rid, run, SMALL + [f for f in step_files(n)])
    print(f"[sync] repo={rid} resume_from={n}", flush=True)
    cmd = [sys.executable, "-m", "task1.train", "--method", a.method, "--out", a.out]
    if n is not None:
        cmd += ["--resume", str(run / str(n))]
    if a.set:
        cmd += ["--set", *a.set]
    stop = threading.Event()

    def loop():
        while not stop.wait(a.interval):
            try:
                sync_once(api, rid, run, a.hf_keep, a.prune_local)
            except Exception as e:
                print("[sync] error (will retry):", e, flush=True)

    t = threading.Thread(target=loop, daemon=True); t.start()
    rc = subprocess.call(cmd)
    stop.set(); t.join()
    if rc == 0:
        sync_once(api, rid, run, a.hf_keep, a.prune_local, force=True)
    print(f"[sync] done rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
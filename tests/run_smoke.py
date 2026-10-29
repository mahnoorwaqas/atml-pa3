"""Run every smoke test:  python -m tests.run_smoke   (works with or without pytest)."""
import importlib, sys, time, traceback

MODULES = ["tests.test_task3", "tests.test_task1", "tests.test_task2"]


def main():
    ok = bad = 0
    for mn in MODULES:
        try:
            mod = importlib.import_module(mn)
        except Exception:
            print(f"[IMPORT FAIL] {mn}"); traceback.print_exc(); bad += 1; continue
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t = time.time()
            try:
                getattr(mod, name)(); ok += 1; print(f"[PASS] {mn}.{name}  ({time.time() - t:.1f}s)")
            except Exception:
                bad += 1; print(f"[FAIL] {mn}.{name}"); traceback.print_exc()
    print(f"\n{ok} passed, {bad} failed"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

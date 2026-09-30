#!/usr/bin/env python3
"""
Run torch_train.py on a Kaggle GPU from this machine, and bring the results home.

Nothing here is needed to train locally - train.py does that on its own, and torch_train.py runs
anywhere torch is installed. This only moves the same work to a free T4: it uploads the pool as a
PRIVATE Kaggle dataset (re-uploading only when the pool has changed), pushes a PRIVATE script that
runs torch_train.py once per requested configuration, waits for it, and copies each out-of-fold
file into captures/heads/, where smooth.py and the label editor read it like any other.

One-time setup: a Kaggle account with a verified phone (for GPUs), the kaggle CLI
(`pip install kaggle`), and an API token in ~/.kaggle/access_token.

    python trainer/kaggle_run.py --export --run "k40_lr head=lr l2=1e-3" \\
                                          --run "k40_mlp16 head=mlp hidden=16 wd=1e-2"
    python trainer/kaggle_run.py --run "k40_mlp32 head=mlp hidden=32" --max-folds 2   # smoke test

--export rebuilds the pool with train.py first; without it the last exported pool is used.
What is uploaded: YAMNet embeddings (numbers, not audio), per-frame labels and fold assignment,
and recording names. No audio leaves the machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "captures" / "kaggle"
DATA_DIR = WORK / "dataset"
POOL = DATA_DIR / "pool.npz"
KERNEL_DIR = WORK / "kernel"
HEADS = ROOT / "captures" / "heads"
DATASET_SLUG = "ads-filter-pool"
KERNEL_SLUG = "ads-filter-train"
MARKER = "# ---- entry point"


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def username() -> str:
    for line in kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("no Kaggle username - is ~/.kaggle/access_token in place?")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_pool(user: str) -> None:
    """Create the private dataset, or add a version if the pool changed since the last upload."""
    stamp = WORK / "pool.sha256"
    digest = sha(POOL)
    if stamp.exists() and stamp.read_text().strip() == digest:
        print("pool unchanged since the last upload - not re-uploading")
        return
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}]}, indent=1))
    exists = "ads-filter-pool" in kaggle("datasets", "list", "--mine", "-s", DATASET_SLUG,
                                         check=False)
    print(f"uploading {POOL.stat().st_size / 1e6:.0f} MB ({'new version' if exists else 'new private dataset'})")
    if exists:
        kaggle("datasets", "version", "-p", str(DATA_DIR), "-m", f"pool {digest[:12]}")
    else:
        kaggle("datasets", "create", "-p", str(DATA_DIR))       # private unless --public
    # Kaggle unpacks and indexes an upload before a kernel can mount it.
    for _ in range(120):
        status = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).strip()
        if "ready" in status.lower():
            break
        time.sleep(10)
    else:
        raise SystemExit(f"dataset never became ready: {status}")
    stamp.write_text(digest)


def parse_run(spec: str) -> dict:
    """'name head=mlp hidden=16 wd=1e-2' -> {'name': ..., 'head': 'mlp', 'hidden': 16, ...}"""
    name, *pairs = spec.split()
    out = {"name": name}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    if "head" not in out:
        raise SystemExit(f"--run {spec!r} needs head=lr or head=mlp")
    return out


def write_kernel(user: str, runs: list[dict], max_folds: int) -> None:
    """torch_train.py, verbatim up to its entry point, plus a runner for these configurations."""
    src = (ROOT / "trainer" / "torch_train.py").read_text(encoding="utf-8")
    body = src[:src.index(MARKER)]
    runner = f'''
# ---- runner written by kaggle_run.py ----
import glob as _glob
_POOL = _glob.glob("/kaggle/input/**/pool.npz", recursive=True)[0]
_RUNS = {json.dumps(runs)}
_summaries = {{}}
for _r in _RUNS:
    _r = dict(_r)
    _name = _r.pop("name")
    _head = _r.pop("head")
    print("=" * 80, flush=True)
    _summaries[_name] = run(_POOL, _head, f"/kaggle/working/{{_name}}.npz", "auto", {max_folds},
                            log=lambda *a: print(*a, flush=True), **_r)
with open("/kaggle/working/summary.json", "w") as _f:
    json.dump(_summaries, _f, indent=1)
'''
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "train_kernel.py").write_text(body + runner, encoding="utf-8")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "train_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [],
        "kernel_sources": []}, indent=1))


def wait_and_fetch(user: str, runs: list[dict], poll: int) -> Path:
    ref = f"{user}/{KERNEL_SLUG}"
    t0 = time.time()
    while True:
        # Only a real answer counts. A laptop closed in a bag comes back with no network, and the
        # client's connection error must not be mistaken for the job failing - the job is on
        # Kaggle's machine and carries on regardless; the thing to do is ask again later.
        reply = kaggle("kernels", "status", ref, check=False)
        status = next((ln.strip() for ln in reply.splitlines() if "has status" in ln), None)
        if status is None:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] no answer from Kaggle "
                  f"(offline?) - retrying", flush=True)
        elif any(w in status.lower() for w in ("complete", "error", "cancel")):
            break
        else:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] {status}", flush=True)
        time.sleep(poll)
    out = WORK / "out" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(out), check=False)
    print(f"{status}\noutput -> {out}")
    if "complete" not in status.lower():
        log = next(out.glob("*.log"), None)
        tail = log.read_text(encoding="utf-8", errors="replace")[-3000:] if log else "(no log)"
        raise SystemExit(f"the Kaggle run did not complete. End of its log:\n{tail}")
    HEADS.mkdir(parents=True, exist_ok=True)
    for r in runs:
        f = out / f"{r['name']}.npz"
        if f.exists():
            shutil.copy2(f, HEADS / f"oof_{r['name']}.npz")
            print(f"  -> captures/heads/oof_{r['name']}.npz")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--run", action="append", required=True, metavar='"NAME head=... k=v"',
                    help="one torch_train.py configuration; repeat for several in one GPU session")
    ap.add_argument("--export", action="store_true",
                    help="rebuild the pool from captures/stitched and captures/sessions first")
    ap.add_argument("--fetch", action="store_true",
                    help="do not push: wait for the job already on Kaggle and download its "
                         "results (give the same --run names it was pushed with)")
    ap.add_argument("--max-folds", type=int, default=0, help="smoke test: this many folds only")
    ap.add_argument("--poll", type=int, default=30, help="seconds between status checks")
    args = ap.parse_args()

    runs = [parse_run(s) for s in args.run]
    if args.fetch:
        # Rejoin a job pushed earlier - after the waiting process died, or the machine slept -
        # without pushing anything, which would replace the job and its results.
        out = wait_and_fetch(username(), runs, args.poll)
        print_summary(out)
        return 0
    if args.export or not POOL.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "trainer" / "train.py"),
                        "--dir", str(ROOT / "captures" / "stitched"),
                        "--dir", str(ROOT / "captures" / "sessions"),
                        "--export", str(POOL)], check=True)
    user = username()
    upload_pool(user)
    write_kernel(user, runs, args.max_folds)
    print(kaggle("kernels", "push", "-p", str(KERNEL_DIR)).strip())
    out = wait_and_fetch(user, runs, args.poll)
    print_summary(out)
    return 0


def print_summary(out: Path) -> None:
    summary = out / "summary.json"
    if summary.exists():
        for name, s in json.loads(summary.read_text()).items():
            print(f"{name:14s} precision {100 * s['precision']:5.1f}%  recall "
                  f"{100 * s['recall']:5.1f}%  {s['folds']} folds in {s['seconds']:.0f}s "
                  f"on {s['device']}")


if __name__ == "__main__":
    raise SystemExit(main())

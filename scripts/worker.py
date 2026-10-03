"""File-queue worker (spec 09 §9.3): claims jobs by atomic rename, one subprocess per slot."""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from dartgxe.paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--slots", type=int, default=3)
ap.add_argument("--prefix", default="", help="only claim jobs whose file name starts with this")
a = ap.parse_args()
J = ROOT / "jobs"
for d in ("pending", "running", "done", "failed", "logs"):
    (J / d).mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[1]


def claim():
    for p in sorted((J / "pending").glob(f"{a.prefix}*.json")):
        dst = J / "running" / p.name
        try:
            os.rename(p, dst)  # atomic on one filesystem: only one worker wins
            return dst
        except FileNotFoundError:
            continue
    return None


def slot(i):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    while (job := claim()) is not None:
        log = J / "logs" / (job.stem + ".log")
        with open(log, "w") as f:
            rc = subprocess.call([sys.executable, str(REPO / "scripts" / "train_stage1.py"), "--job", str(job)],
                                 stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        os.rename(job, J / ("done" if rc == 0 else "failed") / job.name)


ts = [threading.Thread(target=slot, args=(i,)) for i in range(a.slots)]
for t in ts:
    t.start()
    time.sleep(2)
for t in ts:
    t.join()
print("worker idle: no pending jobs")

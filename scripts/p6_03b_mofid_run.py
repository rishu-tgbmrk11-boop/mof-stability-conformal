"""Phase 6 (3): compute MOFid (MOFid-v1 from snurr-group/mofid, commit in notes/phase6.md) for every optimof MOF that has a CIF in the CoRE MOF 2019 ASR archive
(data/raw/coremof2019/cifs; matched through the table's `CoRE_name` column for thermal / solvent MOFs, otherwise `<name>_clean`). Resumable: results are appended to
data/processed/phase6/mofid_results.csv. Each MOF runs in a worker with a hard timeout and a private scratch directory that is removed afterwards.
python -I scripts/p6_03b_mofid_run.py [N_WORKERS]   (run with the mof-stability env active; NOT -I for the mofid import path, the package is pip-installed so -I works too)"""
import csv
import glob
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import time

import pandas as pd

OUT = "data/processed/phase6/mofid_results.csv"
CIFS = "data/raw/coremof2019/cifs/"
TIMEOUT = 300
R = "data/raw/optimof/data/"


def targets():
    th = pd.read_csv(R + "thermal/thermal_all_data.csv"); sv = pd.read_csv(R + "solvent/solvent_all_data.csv"); wt = pd.read_csv(R + "water_and_haz/data.csv")
    core = {}
    for df in (th, sv):
        for r, c in zip(df.refcode, df.CoRE_name):
            core[r] = c
    names = sorted(set(th.refcode) | set(sv.refcode) | set(wt.MOF_name))
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(CIFS + "*/*.cif")}
    out = []
    for n in names:
        c = core.get(n, n + "_clean")
        p = files.get(c) or files.get(n + "_clean") or files.get(n)
        if p:
            out.append((n, p))
    return out


def work(arg):
    name, cif = arg
    from mofid.run_mofid import cif2mofid
    tmp = tempfile.mkdtemp(prefix="mofid_", dir=os.environ.get("MOFID_SCRATCH", None))
    t0 = time.time(); status, mofid, mofkey = "ok", "", ""
    def handler(*a): raise TimeoutError("timeout")
    signal.signal(signal.SIGALRM, handler); signal.alarm(TIMEOUT)
    try:
        so = os.dup(1); se = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY); os.dup2(dn, 1); os.dup2(dn, 2)          # MOFid is chatty
        try:
            r = cif2mofid(cif, output_path=os.path.join(tmp, "out")); mofid, mofkey = r["mofid"], r.get("mofkey", "")
        finally:
            os.dup2(so, 1); os.dup2(se, 2); os.close(dn)
    except TimeoutError:
        status = "timeout"
    except BaseException as e:
        status = f"error:{type(e).__name__}:{str(e)[:80]}"
    finally:
        signal.alarm(0); shutil.rmtree(tmp, ignore_errors=True)
    return dict(name=name, cif=os.path.relpath(cif), status=status, mofid=mofid, mofkey=mofkey, secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    T = targets(); done = set(pd.read_csv(OUT).name) if os.path.exists(OUT) else set()
    todo = [t for t in T if t[0] not in done]
    print(f"{len(T)} MOFs with a CIF; already done {len(done)}; to run {len(todo)}; workers {nw}", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh, mp.Pool(nw) as pool:
        w = csv.DictWriter(fh, fieldnames=["name", "cif", "status", "mofid", "mofkey", "secs"])
        if new: w.writeheader()
        for i, r in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            w.writerow(r); fh.flush()
            if i % 100 == 0: print(f"{i}/{len(todo)} done", flush=True)
    print("DONE", flush=True)

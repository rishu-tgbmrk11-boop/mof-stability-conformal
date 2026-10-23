"""Audit step 1: list the globals every optimof pickle asks for, and whether the restricted loader can load it.
Run as:  python -I scripts/audit_01_scan_pickles.py      (from the repo root)"""
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from src.models import safe_pickle as sp

ROOT = "data/raw/optimof"
files = sorted(glob.glob(f"{ROOT}/data/**/*.pkl", recursive=True) + glob.glob(f"{ROOT}/model/**/*.pkl", recursive=True))
report = {}
for f in files:
    rel = os.path.relpath(f, ROOT)
    magic = open(f, "rb").read(2)
    entry = {"bytes": os.path.getsize(f), "format": "torch-zip" if magic == b"PK" else "pickle/joblib"}
    if entry["format"] == "torch-zip":
        entry["globals_static"] = [f"{m}.{n}" for m, n in sp.scan_torch_zip_globals(f)]
    else:
        seen = []
        try:
            obj = sp.safe_joblib_load(f, seen=seen)
            entry["loaded_type"] = f"{type(obj).__module__}.{type(obj).__name__}"
            entry["blocked"] = None
        except sp.BlockedGlobalError as e:
            entry["blocked"] = str(e)
            entry["loaded_type"] = None
        except Exception as e:
            entry["blocked"] = f"load error: {type(e).__name__}: {str(e)[:150]}"
        entry["n_globals_requested"] = len(set(seen))
        entry["modules_requested"] = sorted({m.split(".")[0] for m, _ in seen})
    report[rel] = entry
json.dump(report, open("data/processed/audit/pickle_scan.json", "w"), indent=1)
for k, v in report.items():
    print(f"\n{k}  [{v['format']}, {v['bytes']:,} bytes]")
    if v["format"] == "torch-zip":
        for g in v["globals_static"]:
            print("   global:", g)
    else:
        print("   loaded_type:", v["loaded_type"], "| modules:", v["modules_requested"], "| blocked:", v["blocked"])

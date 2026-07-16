#!/usr/bin/env python3
"""Check if a Frigate blue/green training run has results.csv and best.pt weights."""
from pathlib import Path
import csv, json
root=Path('/mnt/user/media/frigate_custom_model/runs/blue_package_plus_local_20260706_1305')
p=root/'results.csv'
w=root/'weights/best.pt'
out={'run_exists':root.exists(),'results_exists':p.exists(),'best_pt_exists':w.exists(),'best_pt_bytes':w.stat().st_size if w.exists() else 0}
if p.exists():
    rows=list(csv.DictReader(p.read_text().splitlines()))
    out['rows']=len(rows)
    out['last_metrics']=rows[-1] if rows else {}
print(json.dumps(out, indent=2, sort_keys=True))

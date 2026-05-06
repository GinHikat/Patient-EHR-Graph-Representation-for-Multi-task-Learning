"""
Check timeline length distribution across all patients.
Reads from Timelines/{pid}_meta.json files.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm
import os, sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set project root relative to this script
# modules/test/event_length.py -> modules/test -> modules -> project_root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

data_dir = os.getenv('DATA_DIR')

if data_dir is None:
    # Fallback to local data directory if DATA_DIR is not set
    data_dir = os.path.join(project_root, 'data')
    print(f"Warning: DATA_DIR not found in environment. Using fallback: {data_dir}")

base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

TIMELINE_DIR = os.path.join(data_dir, "Timelines")   # update if needed
PATIENTS_TXT = os.path.join(downstream_data_path, "patients.txt")

# ── Load patient IDs ──────────────────────────────────────────────────────
with open(PATIENTS_TXT) as f:
    all_pids = [line.strip() for line in f if line.strip()]

print(f"Total patients: {len(all_pids):,}")

# ── Collect lengths ───────────────────────────────────────────────────────
lengths       = []
missing       = []
event_types   = {}   # count by event type across all patients

for pid in tqdm(all_pids[:20000], desc="Reading timelines"):
    meta_path = Path(TIMELINE_DIR) / f"{pid}_meta.json"
    if not meta_path.exists():
        missing.append(pid)
        continue

    with open(meta_path) as f:
        meta = json.load(f)

    lengths.append(len(meta))

    for entry in meta:
        t = entry.get("type", "unknown")
        event_types[t] = event_types.get(t, 0) + 1

# ── Summary stats ─────────────────────────────────────────────────────────
lengths = np.array(lengths)

print(f"\n── Timeline length distribution ({len(lengths):,} patients) ──")
print(f"  Min:    {lengths.min()}")
print(f"  p5:     {np.percentile(lengths,  5):.0f}")
print(f"  p25:    {np.percentile(lengths, 25):.0f}")
print(f"  Median: {np.median(lengths):.0f}")
print(f"  Mean:   {lengths.mean():.1f}")
print(f"  p75:    {np.percentile(lengths, 75):.0f}")
print(f"  p90:    {np.percentile(lengths, 90):.0f}")
print(f"  p95:    {np.percentile(lengths, 95):.0f}")
print(f"  p99:    {np.percentile(lengths, 99):.0f}")
print(f"  Max:    {lengths.max()}")

print(f"\n── Event type breakdown (across all patients) ──")
total_events = sum(event_types.values())
for etype, count in sorted(event_types.items(), key=lambda x: -x[1]):
    print(f"  {etype:<20s}  {count:>10,}  ({100*count/total_events:.1f}%)")

print(f"\n── Missing timelines: {len(missing):,} patients ──")
if missing[:5]:
    print(f"  First few: {missing[:5]}")

# ── Bucket distribution ───────────────────────────────────────────────────
print(f"\n── Length buckets ──")
buckets = [0, 10, 25, 50, 100, 200, 500, 99999]
labels  = ["<10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
for i, label in enumerate(labels):
    lo, hi  = buckets[i], buckets[i+1]
    count   = ((lengths >= lo) & (lengths < hi)).sum()
    print(f"  {label:<10s}  {count:>7,}  ({100*count/len(lengths):.1f}%)")
 
p95 = np.percentile(lengths, 95)
p99 = np.percentile(lengths, 99)
 
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Patient Timeline Length Distribution", fontsize=14, fontweight="bold")
 
# Full distribution (log y-scale)
ax.hist(lengths, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
ax.axvline(np.median(lengths), color="orange",  linestyle="--", linewidth=1.5, label=f"Median={np.median(lengths):.0f}")
ax.axvline(p95,                color="red",     linestyle="--", linewidth=1.5, label=f"p95={p95:.0f}")
ax.axvline(p99,                color="darkred", linestyle="--", linewidth=1.5, label=f"p99={p99:.0f}")
ax.set_xlabel("Number of events")
ax.set_ylabel("Number of patients")
ax.set_title("Full distribution (log scale)")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
 
plt.tight_layout()
out_path = os.path.join(base_data_dir, 'images', 'event_length_distribution.png')
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nHistogram saved → {out_path}")
plt.close()
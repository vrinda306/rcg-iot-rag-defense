"""CICIoT2023 sample: seed 42, per-class cap, sampled from train.csv only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ATTACK_CAP = 20_000
NORMAL_CAP = 140_000      # change here if you and your teammate agree on another number
LABEL = "label"
NORMAL_RAW = "BenignTraffic"

ROOT = Path(r"C:\Users\nishi\OneDrive\Desktop\rcg-iot-rag-defense")
SRC = ROOT / "data" / "raw" / "ciciot" / "CICIOT23" / "train" / "train.csv"
OUT = ROOT / "data" / "processed" / "sample_ciciot_v1.parquet"
META = OUT.with_suffix(".meta.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Pass 1: read only the label column and choose which rows to keep
labels = pd.read_csv(SRC, usecols=[LABEL])[LABEL]
rng = np.random.default_rng(SEED)
keep, full_counts, sample_counts = [], {}, {}
for cls, idx in labels.groupby(labels).indices.items():
    cap = NORMAL_CAP if cls == NORMAL_RAW else ATTACK_CAP
    chosen = idx if len(idx) <= cap else rng.choice(idx, size=cap, replace=False)
    full_counts[cls] = int(len(idx))
    sample_counts[cls] = int(len(chosen))
    keep.append(chosen)
keep = np.sort(np.concatenate(keep))
del labels

# Pass 2: read the CSV in chunks and keep only the chosen rows
parts, start = [], 0
for chunk in pd.read_csv(SRC, chunksize=500_000):
    end = start + len(chunk)
    rows = keep[(keep >= start) & (keep < end)] - start
    parts.append(chunk.iloc[rows])
    start = end
sample = pd.concat(parts, ignore_index=True)
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

# Check the sample matches the plan
actual = sample[LABEL].value_counts().to_dict()
assert actual == sample_counts, "Sample counts do not match the plan"

features = [c for c in sample.columns if c != LABEL]
constant = [c for c in features if sample[c].nunique() == 1]

sample.to_parquet(OUT, index=False)
meta = {
    "source_file": str(SRC.relative_to(ROOT)),
    "seed": SEED,
    "attack_cap_per_class": ATTACK_CAP,
    "normal_cap": NORMAL_CAP,
    "normal_raw_label": NORMAL_RAW,
    "rows": int(len(sample)),
    "n_classes": int(sample[LABEL].nunique()),
    "features": features,
    "excluded_columns": [LABEL],
    "constant_columns_in_sample": constant,
    "class_counts_full_train": full_counts,
    "class_counts_sample": sample_counts,
}
META.write_text(json.dumps(meta, indent=2))
print("Rows:", len(sample), "| classes:", sample[LABEL].nunique())
print("Constant columns in sample:", constant)
print("Saved:", OUT)
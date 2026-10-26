"""Create a reproducible, class-capped sample of Edge-IIoTset (numeric features only).

Design notes (for the viva):
- The source file is sorted by class, one class per row group, so we never read "the first N rows".
- Pass 1 counts rows per (row group, class). Pass 2 reads one group at a time.
- Each class gets min(total, cap) rows, spread across its row groups in proportion to group size.
- Final shuffle with a fixed seed makes the output order independent of the source layout.
- src_row_group is stored for diagnostics only. It leaks the label and is never a feature.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
SRC = Path("data/edgeiiotset_scenarios_v1_ml_ready.parquet")
OUT = Path("data/processed/sample_v1.parquet")
META = Path("data/processed/sample_v1.meta.json")
CAP_ATTACK = 20_000
CAP_NORMAL = 140_000

NUMERIC = [
    "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le", "icmp.unused",
    "http.content_length", "http.tls_port", "tcp.ack", "tcp.ack_raw", "tcp.checksum",
    "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.seq",
    "udp.stream", "udp.time_delta", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
    "mqtt.msgtype", "mqtt.proto_len", "mqtt.topic_len", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
]
LABELS = ["Attack_label", "Attack_type"]


def main():
    # ---- Sanity checks before doing any work ----
    if not SRC.exists():
        raise SystemExit(f"Source file not found: {SRC.resolve()}")
    pf = pq.ParquetFile(SRC)

    assert len(NUMERIC) == 37 and len(set(NUMERIC)) == 37, "Expected 37 unique feature names"
    assert "src_row_group" not in NUMERIC, "src_row_group must never be a feature"

    missing_cols = [c for c in NUMERIC + LABELS if c not in pf.schema_arrow.names]
    if missing_cols:
        raise SystemExit(f"Columns not found in file: {missing_cols}")

    schema = pf.schema_arrow
    non_numeric = [
        c for c in NUMERIC
        if not (pa.types.is_integer(schema.field(c).type)
                or pa.types.is_floating(schema.field(c).type)
                or pa.types.is_boolean(schema.field(c).type))
    ]
    if non_numeric:
        raise SystemExit(f"Non-numeric feature columns found: {non_numeric}")

    # ---- Pass 1: rows per (row group, class) ----
    counts = {}
    for g in range(pf.num_row_groups):
        vc = pf.read_row_group(g, columns=["Attack_type"]).column(0).to_pandas().value_counts()
        for cls, n in vc.items():
            counts[(g, cls)] = int(n)

    totals = {}
    for (g, cls), n in counts.items():
        totals[cls] = totals.get(cls, 0) + n

    # ---- Allocation: proportional across groups, largest-remainder rounding ----
    alloc = {}
    for cls, total in totals.items():
        target = min(total, CAP_NORMAL if cls == "Normal" else CAP_ATTACK)
        keys = [k for k in counts if k[1] == cls]
        exact = {k: target * counts[k] / total for k in keys}
        base = {k: int(np.floor(v)) for k, v in exact.items()}
        leftover = target - sum(base.values())
        for k in sorted(keys, key=lambda k: exact[k] - base[k], reverse=True)[:leftover]:
            base[k] += 1
        assert sum(base.values()) == target
        assert all(base[k] <= counts[k] for k in keys)
        alloc.update(base)

    # ---- Pass 2: read one group at a time and sample ----
    rng = np.random.default_rng(SEED)
    parts = []
    for g in range(pf.num_row_groups):
        need = {cls: k for (gg, cls), k in alloc.items() if gg == g and k > 0}
        if not need:
            continue
        df = pf.read_row_group(g, columns=NUMERIC + LABELS).to_pandas()
        for cls, k in need.items():
            sub = df[df["Attack_type"] == cls]
            idx = np.sort(rng.choice(len(sub), size=k, replace=False))
            part = sub.iloc[idx].copy()
            # DIAGNOSTIC ONLY. Each group holds one class, so this column leaks the label.
            # It must NEVER be used as a model feature.
            part["src_row_group"] = g
            parts.append(part)

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT, index=False)

    meta = {
        "source_file": str(SRC),
        "seed": SEED,
        "cap_per_attack_class": CAP_ATTACK,
        "cap_normal": CAP_NORMAL,
        "n_rows": int(len(sample)),
        "numeric_features": NUMERIC,
        "excluded_string_features": [
            "http.request.method", "http.referer", "http.request.version", "http.response",
            "dns.qry.name", "mqtt.conack.flags", "mqtt.msg_decoded_as",
            "mqtt.protoname", "mqtt.topic",
        ],
        "class_counts_sample": {k: int(v) for k, v in sample["Attack_type"].value_counts().items()},
        "class_counts_full": {k: int(v) for k, v in totals.items()},
        "diagnostic_only_columns": ["src_row_group"],
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", OUT)
    print("Shape:", sample.shape)
    print("\nSample class counts:")
    print(sample["Attack_type"].value_counts().to_string())
    print("\nBinary label counts:")
    print(sample["Attack_label"].value_counts().sort_index().to_string())
    print("\nTotal missing values:", int(sample[NUMERIC].isna().sum().sum()))
    print("Exact duplicate rows (features + labels):",
          int(sample.duplicated(subset=NUMERIC + LABELS).sum()))


if __name__ == "__main__":
    main()
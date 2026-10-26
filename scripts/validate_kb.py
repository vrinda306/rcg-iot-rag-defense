"""Validate a knowledge-base JSONL file against a labels YAML file.

Usage:
    python scripts/validate_kb.py data/kb/edgeiiotset_kb.jsonl configs/labels.yaml

Checks format and coverage only. It CANNOT check that a URL is real or that the
description matches the source. A human must open each URL and read it.
"""
import json
import sys
from pathlib import Path

import yaml

REQUIRED = ["id", "label", "title", "description", "mitigation", "source_name",
            "source_url", "source_type", "trust_tier", "retrieved_date"]
TIERS = {1, 2, 3, None}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/validate_kb.py <kb.jsonl> <labels.yaml>")
    kb_path, labels_path = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (kb_path, labels_path):
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    labels = set(yaml.safe_load(labels_path.read_text(encoding="utf-8"))["labels"])
    errors, warnings, entries = [], [], []

    for n, line in enumerate(kb_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as ex:
            errors.append(f"line {n}: not valid JSON ({ex.msg})")
            continue
        entries.append((n, e))

    seen_ids = set()
    for n, e in entries:
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            errors.append(f"line {n}: missing fields {missing}")
            continue
        if e["id"] in seen_ids:
            errors.append(f"line {n}: duplicate id {e['id']}")
        seen_ids.add(e["id"])
        if e["label"] not in labels:
            errors.append(f"line {n}: label '{e['label']}' is not in the labels file (check spelling/capitals)")
        if len(str(e["description"]).strip()) < 40:
            errors.append(f"line {n} ({e['label']}): description is empty or too short")
        if e["trust_tier"] not in TIERS:
            errors.append(f"line {n} ({e['label']}): trust_tier must be 1, 2, 3 or null")
        url = str(e["source_url"]).strip()
        if not url:
            warnings.append(f"line {n} ({e['label']}): no source_url (flagged, not an error)")
        elif not url.startswith("https://"):
            errors.append(f"line {n} ({e['label']}): source_url must start with https://")
        if url and e["trust_tier"] is None:
            warnings.append(f"line {n} ({e['label']}): has a URL but trust_tier is null")

    covered = {e["label"] for _, e in entries if "label" in e}
    absent = sorted(labels - covered)
    if absent:
        errors.append(f"labels with NO entry ({len(absent)}): {absent}")

    print(f"Entries read: {len(entries)}   Labels required: {len(labels)}   Labels covered: {len(covered & labels)}")
    print(f"\nERRORS ({len(errors)}):")
    for m in errors:
        print("  -", m)
    print(f"\nWARNINGS ({len(warnings)}):")
    for m in warnings:
        print("  -", m)
    print("\nRESULT:", "PASS (format only)" if not errors else "FAIL")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
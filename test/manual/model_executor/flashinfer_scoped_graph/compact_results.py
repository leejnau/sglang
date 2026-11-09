"""Publish timing samples and counted lookups; retain raw evidence separately."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def compact(path):
    raw = path.read_bytes()
    row = json.loads(raw)
    row["raw_file_sha256"] = hashlib.sha256(raw).hexdigest()
    row["raw_file_bytes"] = len(raw)
    selections = row.pop("selections", [])
    counts = Counter(json.dumps(s, sort_keys=True) for s in selections)
    row["selection_record_count"] = len(selections)
    row["counted_selections"] = [
        dict(json.loads(key), count=count) for key, count in sorted(counts.items())
    ]
    for capture in row.get("captures", []):
        coverage = capture.get("coverage")
        if coverage is not None:
            records = coverage.pop("records", [])
            coverage["record_count"] = len(records)
            coverage["records_sha256"] = hashlib.sha256(
                json.dumps(records, sort_keys=True).encode()
            ).hexdigest()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.root.glob("matrix-*.json")):
        row = compact(path)
        (args.output / path.name).write_text(json.dumps(row, indent=2) + "\n")
    for name in ("cohort-analysis.json", "cross-process-numerics.json"):
        (args.output / name).write_bytes((args.root / name).read_bytes())


if __name__ == "__main__":
    main()

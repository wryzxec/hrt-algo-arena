"""Validate submission artifacts without opening any in-sample targets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arena.data import load_sessions


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_predictions(frame, expected_ids, id_column="headline_id"):
    """Check exact schema and identity; row order need not match expected_ids."""
    if list(frame.columns) != [id_column, "pred"]:
        raise ValueError(f"Expected exactly {id_column},pred columns in that order")
    expected = pd.Index(expected_ids)
    ids = frame[id_column]
    if not expected.is_unique or expected.hasnans:
        raise ValueError("Expected IDs must be unique and nonmissing")
    if not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any():
        raise ValueError("Session IDs must be nonmissing integers")
    if not ids.is_unique or len(frame) != len(expected) or set(ids) != set(expected):
        raise ValueError("Session IDs must match the expected set exactly once")
    if not pd.api.types.is_numeric_dtype(frame.pred.dtype):
        raise ValueError("Predictions must be numeric")
    pred = frame.pred.to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Predictions must all be finite")
    return frame.set_index(id_column).loc[expected, "pred"].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    oos, _ = load_sessions("oos", root=args.root)
    if len(oos) != 10000:
        raise ValueError("Competition requires exactly 10,000 OOS sessions")
    frame = pd.read_csv(args.csv)
    validate_predictions(frame, oos.index)
    print(json.dumps({"file": args.csv, "rows": len(frame),
                      "sha256": file_sha256(args.csv), "valid": True}, indent=2))


if __name__ == "__main__":
    main()

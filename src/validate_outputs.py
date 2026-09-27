import csv
from pathlib import Path

import pandas as pd


MATCH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")


def _split_ids(value):
    values = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise ValueError("duplicate IDs inside an output row")
    return values


def _read_output(path, expected_columns):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"{path} header must be {expected_columns}")
        rows = {}
        for row in reader:
            source_id = row["source1_entity_id"]
            if not source_id or source_id in rows:
                raise ValueError(f"duplicate or empty Source 1 ID in {path}: {source_id!r}")
            rows[source_id] = _split_ids(row[expected_columns[1]])
    return rows


def validate_outputs(data_root, output_dir="output", chunk_size=100_000):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    train_or_test = next(data_root.rglob("test_source1.tsv"), None)
    test_source2 = next(data_root.rglob("test_source2.tsv"), None)
    test_source3 = next(data_root.rglob("test_source3.tsv"), None)
    if not all((train_or_test, test_source2, test_source3)):
        raise FileNotFoundError("test_source1.tsv, test_source2.tsv, and test_source3.tsv are required")

    expected_s1 = []
    for chunk in pd.read_csv(train_or_test, sep="\t", dtype=str, keep_default_na=False,
                             usecols=["entity_id"], chunksize=chunk_size):
        expected_s1.extend(chunk["entity_id"].tolist())
    expected_s1_set = set(expected_s1)
    if len(expected_s1_set) != len(expected_s1):
        raise ValueError("test Source 1 contains duplicate IDs")

    matching = _read_output(output_dir / "matching_results.tsv", MATCH_COLUMNS)
    candidates = _read_output(output_dir / "candidate_pairs.tsv", CANDIDATE_COLUMNS)
    if set(matching) != expected_s1_set or set(candidates) != expected_s1_set:
        raise ValueError("outputs must contain exactly one row for every test Source 1 ID")

    used_target_ids = set()
    for source_id, values in candidates.items():
        used_target_ids.update(values)
        matched = set(matching[source_id])
        if not matched.issubset(values):
            raise ValueError(f"matches are not a subset of candidates for {source_id}")
        if any(not value.startswith(("S2-", "S3-")) for value in matched):
            raise ValueError(f"invalid match prefix for {source_id}")

    valid_target_ids = set()
    for path in (test_source2, test_source3):
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=["entity_id"], chunksize=chunk_size):
            valid_target_ids.update(set(chunk["entity_id"]) & used_target_ids)
    invalid = used_target_ids - valid_target_ids
    if invalid:
        raise ValueError(f"outputs contain target IDs absent from test data: {next(iter(invalid))}")

    return {
        "status": "PASS",
        "test_source1_rows": len(expected_s1),
        "candidate_rows": len(candidates),
        "matching_rows": len(matching),
        "candidate_target_ids": len(used_target_ids),
        "matched_target_ids": sum(len(values) for values in matching.values()),
    }

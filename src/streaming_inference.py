import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .candidate_benchmark import _blocking_frame, _stream_source_candidates
from .data_loader import discover_files
from .features import make_feature_matrix
from .normalization import add_normalized


COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _load_candidate_rows(path, wanted, chunk_size):
    parts = []
    remaining = set(wanted)
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              usecols=COLUMNS, chunksize=chunk_size):
        found = chunk[chunk["entity_id"].isin(remaining)]
        if len(found):
            parts.append(found)
            remaining.difference_update(found["entity_id"])
            if not remaining:
                break
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return add_normalized(pd.concat(parts, ignore_index=True))


def _score_source(model, s1, target_path, candidate_map, threshold, chunk_size):
    candidate_ids = {target_id for ids in candidate_map.values() for target_id in ids}
    target = _load_candidate_rows(target_path, candidate_ids, chunk_size)
    target = target.drop_duplicates("entity_id").reset_index(drop=True)
    target_index = {value: index for index, value in target["entity_id"].items()}
    indexed_candidates = {
        s1_index: [target_index[target_id] for target_id in target_ids if target_id in target_index]
        for s1_index, target_ids in candidate_map.items()
    }
    features, pairs = make_feature_matrix(s1, target, indexed_candidates)
    probabilities = model.predict_proba(features)[:, 1] if len(features) else np.array([])
    candidates = {entity_id: [] for entity_id in s1["entity_id"]}
    matches = {entity_id: [] for entity_id in s1["entity_id"]}
    for (s1_index, target_index_value), probability in zip(pairs, probabilities):
        source_id = s1.loc[s1_index, "entity_id"]
        target_id = target.loc[target_index_value, "entity_id"]
        candidates[source_id].append(target_id)
        if probability >= threshold:
            matches[source_id].append(target_id)
    return candidates, matches, len(features)


def run_streaming_inference(cfg, model_pack, batch_size=50_000, chunk_size=100_000, sample_s1=None):
    paths = discover_files(cfg.data_root)
    s1_path = paths["test_source1"]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    model = model_pack["model"]
    threshold = float(model_pack["threshold"])
    totals = Counter()
    first_batch = True
    reader = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False,
                         usecols=COLUMNS, chunksize=batch_size, nrows=sample_s1)
    for s1_raw in reader:
        s1 = add_normalized(s1_raw.reset_index(drop=True))
        cm2, _ = _stream_source_candidates(paths["test_source2"], s1, cfg.max_candidates_per_s1, chunk_size)
        cm3, _ = _stream_source_candidates(paths["test_source3"], s1, cfg.max_candidates_per_s1, chunk_size)
        candidates2, matches2, feature_count2 = _score_source(
            model, s1, paths["test_source2"], cm2, threshold, chunk_size
        )
        candidates3, matches3, feature_count3 = _score_source(
            model, s1, paths["test_source3"], cm3, threshold, chunk_size
        )
        with candidate_path.open("a", encoding="utf-8", newline="") as candidate_file, matching_path.open(
            "a", encoding="utf-8", newline=""
        ) as matching_file:
            candidate_writer = csv.DictWriter(candidate_file, fieldnames=["source1_entity_id", "candidate_entity_ids"], delimiter="\t")
            matching_writer = csv.DictWriter(matching_file, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter="\t")
            if first_batch:
                candidate_writer.writeheader()
                matching_writer.writeheader()
                first_batch = False
            for source_id in s1["entity_id"]:
                candidate_ids = sorted(set(candidates2[source_id] + candidates3[source_id]))
                matched_ids = sorted(set(matches2[source_id] + matches3[source_id]))
                if not set(matched_ids).issubset(candidate_ids):
                    raise ValueError(f"model prediction escaped candidate set for {source_id}")
                candidate_writer.writerow({"source1_entity_id": source_id, "candidate_entity_ids": ",".join(candidate_ids)})
                matching_writer.writerow({"source1_entity_id": source_id, "matched_entity_ids": ",".join(matched_ids)})
                totals["source1_rows"] += 1
                totals["candidate_rows"] += len(candidate_ids)
                totals["matched_rows"] += len(matched_ids)
                totals["zero_match_s1"] += not matched_ids
                totals["multi_match_s1"] += len(matched_ids) > 1
                totals["feature_rows"] += feature_count2 + feature_count3
    (cfg.report_dir / "streaming_inference.json").write_text(json.dumps(dict(totals), indent=2), encoding="utf-8")
    return dict(totals)

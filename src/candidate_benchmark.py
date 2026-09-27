import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import build_target_index, generate_candidates
from .data_loader import discover_files, load_tsv
from .ground_truth import parse_ground_truth
from .normalization import normalize_address, normalize_country, normalize_name


RULES = (
    "exact_name", "exact_address", "country_name", "country_address",
    "name_token", "address_token", "prefix_name", "prefix_address",
    "exact_compact_name", "compact_name_prefix", "exact_compact_address",
    "compact_address_prefix", "significant_name_token",
    "significant_name_token_pair", "address_number", "address_number_token",
)
BUCKETS = ("0", "1", "2", "3", "4", "5+")


def _bucket(count):
    return str(count) if count < 5 else "5+"


def _blocking_frame(frame):
    out = frame.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_country)
    return out


def _sample_ids(ground_truth, sample_size, random_state):
    counts = ground_truth["matched_entity_ids"].map(
        lambda value: 0 if not str(value).strip() else len(
            {item.strip() for item in str(value).split(",") if item.strip()}
        )
    )
    ground_truth = ground_truth.assign(_bucket=counts.map(_bucket))
    base = sample_size // 10
    requested = {bucket: base for bucket in BUCKETS[:5]}
    requested["5+"] = sample_size - sum(requested.values())
    chosen = []
    for bucket in BUCKETS:
        rows = ground_truth.loc[ground_truth["_bucket"] == bucket]
        take = min(requested[bucket], len(rows))
        if take:
            chosen.append(rows.sample(n=take, random_state=random_state))
    selected = pd.concat(chosen, ignore_index=True) if chosen else ground_truth.iloc[:0]
    if len(selected) < sample_size:
        remaining = ground_truth.loc[~ground_truth["source1_entity_id"].isin(selected["source1_entity_id"])]
        extra = remaining.sample(n=min(sample_size - len(selected), len(remaining)), random_state=random_state)
        selected = pd.concat([selected, extra], ignore_index=True)
    return selected, counts


def _load_rows_by_id(path, ids, chunk_size=100_000):
    parts = []
    wanted = set(ids)
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
        part = chunk[chunk["entity_id"].isin(wanted)]
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    out = pd.concat(parts, ignore_index=True)
    return out.set_index("entity_id").loc[list(ids)].reset_index()


def _truth_by_id(ground_truth):
    return parse_ground_truth(ground_truth)


def _empty_bucket_metrics():
    return {bucket: {"s1_rows": 0, "true_ids": 0, "recovered_ids": 0, "recall": None} for bucket in BUCKETS}


def _recall_metrics(sample, truth, s2, s3, cm2, cm3, provenance):
    id2s2 = s2["entity_id"].to_dict()
    id2s3 = s3["entity_id"].to_dict()
    s2_index_by_id = {value: key for key, value in id2s2.items()}
    s3_index_by_id = {value: key for key, value in id2s3.items()}
    totals = Counter()
    recovered = Counter()
    rule_counts = Counter()
    rule_recovered = Counter()
    candidate_counts = []
    singleton_counts = []
    missed = []

    for index, row in sample.iterrows():
        s1_id = row["entity_id"]
        true_ids = set(truth.get(s1_id, []))
        candidates2 = {id2s2[i] for i in cm2.get(index, [])}
        candidates3 = {id2s3[i] for i in cm3.get(index, [])}
        candidates = candidates2 | candidates3
        candidate_counts.append(len(candidates))
        if not true_ids:
            singleton_counts.append(len(candidates))
        bucket = _bucket(len(true_ids))
        totals[bucket] += len(true_ids)
        recovered[bucket] += len(true_ids & candidates)
        for source_map in (cm2, cm3):
            for target_index in source_map.get(index, []):
                for rule in provenance.get(source_map is cm3, {}).get(index, {}).get(target_index, ()):
                    rule_counts[rule] += 1
        for true_id in true_ids:
            if true_id.startswith("S2-"):
                target_index = s2_index_by_id.get(true_id)
                source_provenance = provenance[False].get(index, {})
            else:
                target_index = s3_index_by_id.get(true_id)
                source_provenance = provenance[True].get(index, {})
            for rule in source_provenance.get(target_index, ()):
                rule_recovered[rule] += 1
        for true_id in true_ids - candidates:
            missed.append((s1_id, true_id, "s2" if true_id.startswith("S2-") else "s3"))

    nonzero_total = sum(totals.values())
    nonzero_recovered = sum(recovered.values())
    s2_total = sum(value for bucket, value in totals.items() for _, target in [(bucket, None)] if False)
    s2_total = s2_recovered = s3_total = s3_recovered = 0
    for index, row in sample.iterrows():
        true_ids = set(truth.get(row["entity_id"], []))
        candidate_ids = {
            id2s2[i] for i in cm2.get(index, [])
        } | {id2s3[i] for i in cm3.get(index, [])}
        for target_id in true_ids:
            if target_id.startswith("S2-"):
                s2_total += 1
                s2_recovered += target_id in candidate_ids
            elif target_id.startswith("S3-"):
                s3_total += 1
                s3_recovered += target_id in candidate_ids

    by_bucket = _empty_bucket_metrics()
    for bucket in BUCKETS:
        rows = sum(_bucket(len(truth.get(row["entity_id"], []))) == bucket for _, row in sample.iterrows())
        by_bucket[bucket] = {
            "s1_rows": rows,
            "true_ids": totals[bucket],
            "recovered_ids": recovered[bucket],
            "recall": recovered[bucket] / totals[bucket] if totals[bucket] else None,
        }
    return {
        "overall_recall": nonzero_recovered / nonzero_total if nonzero_total else None,
        "s2_recall": s2_recovered / s2_total if s2_total else None,
        "s3_recall": s3_recovered / s3_total if s3_total else None,
        "true_ids": nonzero_total,
        "recovered_ids": nonzero_recovered,
        "missed_ids": nonzero_total - nonzero_recovered,
        "by_match_count": by_bucket,
        "average_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "p90_candidates": float(np.percentile(candidate_counts, 90)) if candidate_counts else 0.0,
        "p95_candidates": float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0,
        "p99_candidates": float(np.percentile(candidate_counts, 99)) if candidate_counts else 0.0,
        "max_candidates": max(candidate_counts, default=0),
        "rule_candidate_counts": {rule: rule_counts[rule] for rule in RULES},
        "rule_true_id_recall": {
            rule: rule_recovered[rule] / nonzero_total if nonzero_total else None
            for rule in RULES
        },
        "singleton": {
            "s1_rows": len(singleton_counts),
            "zero_candidate_rows": sum(count == 0 for count in singleton_counts),
            "zero_candidate_percent": 100.0 * sum(count == 0 for count in singleton_counts) / len(singleton_counts) if singleton_counts else 0.0,
            "average_candidates": float(np.mean(singleton_counts)) if singleton_counts else 0.0,
            "median_candidates": float(np.median(singleton_counts)) if singleton_counts else 0.0,
        },
        "missed": missed,
    }


def _diagnose(s1_row, target_row):
    if s1_row["country_norm"] != target_row["country_norm"]:
        return "country_issue"
    if not s1_row["address_norm"] or not target_row["address_norm"]:
        return "missing_address"
    name_tokens = set(s1_row["name_norm"].split()) & set(target_row["name_norm"].split())
    address_tokens = set(s1_row["address_norm"].split()) & set(target_row["address_norm"].split())
    if not name_tokens and not address_tokens:
        return "no_shared_name_or_address_token"
    if not name_tokens:
        return "no_shared_name_token"
    if not address_tokens:
        return "no_shared_address_token"
    return "prefix_or_combination_gap"


def run_benchmark(data_root, report_dir, sample_size=5000, budgets=(50, 80, 120, 160), random_state=42):
    paths = discover_files(Path(data_root))
    ground_truth = load_tsv(paths["train_ground_truth"])
    selected_gt, _ = _sample_ids(ground_truth, sample_size, random_state)
    selected_ids = selected_gt["source1_entity_id"].tolist()
    s1 = _load_rows_by_id(paths["train_source1"], selected_ids)
    s2 = load_tsv(paths["train_source2"])
    s3 = load_tsv(paths["train_source3"])
    s1 = _blocking_frame(s1).reset_index(drop=True)
    s2 = _blocking_frame(s2).reset_index(drop=True)
    s3 = _blocking_frame(s3).reset_index(drop=True)
    truth = _truth_by_id(selected_gt)
    s2_index = build_target_index(s2)
    s3_index = build_target_index(s3)
    needed_target_ids = {target_id for ids in truth.values() for target_id in ids}
    target_rows = pd.concat([
        s2[s2["entity_id"].isin(needed_target_ids)],
        s3[s3["entity_id"].isin(needed_target_ids)],
    ])
    target_by_id = target_rows.set_index("entity_id").to_dict(orient="index")
    results = {}
    missed_rows = []

    for budget in budgets:
        cm2, prov2 = generate_candidates(s1, s2, max_candidates=budget, index=s2_index, return_provenance=True)
        cm3, prov3 = generate_candidates(s1, s3, max_candidates=budget, index=s3_index, return_provenance=True)
        # Keep provenance source-specific while preserving the evaluator's compact interface.
        provenance = {True: prov3, False: prov2}
        metrics = _recall_metrics(s1, truth, s2, s3, cm2, cm3, provenance)
        for s1_id, target_id, source in metrics.pop("missed")[:500]:
            target_row = target_by_id.get(target_id)
            s1_row = s1.loc[s1["entity_id"] == s1_id].iloc[0]
            if target_row is not None:
                missed_rows.append({
                    "budget": budget, "source1_entity_id": s1_id, "true_target_entity_id": target_id,
                    "source": source, "source1_business_name": s1_row["business_name"],
                    "target_business_name": target_row["business_name"], "source1_business_address": s1_row["business_address"],
                    "target_business_address": target_row["business_address"], "country": s1_row["country"],
                    "attempted_blocking_rules": ";".join(RULES), "generated_blocking_rules": "none",
                    "diagnosis": _diagnose(s1_row, target_row),
                })
        results[str(budget)] = metrics

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output = {"sample_size": len(s1), "requested_sample_size": sample_size, "random_state": random_state,
              "budgets": list(budgets), "results": results}
    (report_dir / "candidate_recall_benchmark.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(missed_rows[:500]).to_csv(report_dir / "missed_candidates_sample.tsv", sep="\t", index=False)
    lines = ["# Candidate Recall Benchmark", "", f"Sampled S1 rows: {len(s1)}", f"Random state: {random_state}", "",
             "| Candidate budget | Avg candidates | Overall recall | S2 recall | S3 recall |", "|---:|---:|---:|---:|---:|"]
    for budget in budgets:
        metric = results[str(budget)]
        lines.append(f"| {budget} | {metric['average_candidates']:.2f} | {metric['overall_recall']:.4f} | {metric['s2_recall']:.4f} | {metric['s3_recall']:.4f} |")
    lines.extend(["", "Zero-match entities are excluded from recall and reported under `singleton` in the JSON.", "", "Detailed metrics are in `candidate_recall_benchmark.json`."])
    (report_dir / "candidate_recall_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    return output
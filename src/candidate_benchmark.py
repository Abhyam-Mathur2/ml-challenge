import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import (
    EXACT_POSTING_LIMIT,
    POSTING_LIMIT,
    _address_numbers,
    _compact,
    _compact_prefixes,
    _prefixes,
    _significant_address_tokens,
    _significant_name_tokens,
    _tokens,
    generate_candidates,
)
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


class _StreamingTargetIndex:
    """TargetIndex-compatible postings built in global target row order."""

    def __init__(self, s1, prefix_length=4):
        self.prefix_length = prefix_length
        self.posting_limit = POSTING_LIMIT
        names = s1["name_norm"]
        addresses = s1["address_norm"]
        countries = s1["country_norm"]
        self._needed = {name: set() for name in (
            "exact_name", "exact_address", "exact_name_country", "exact_address_country",
            "exact_compact_name", "exact_compact_address", "name_token", "country_name_token",
            "significant_name_token", "significant_name_token_pair", "address_token",
            "country_address_token", "address_number", "address_number_token", "name_prefix",
            "address_prefix", "compact_name_prefix", "compact_address_prefix")}
        for name, address, country in zip(names, addresses, countries):
            if name:
                self._needed["exact_name"].add(name)
                self._needed["exact_name_country"].add((name, country))
                self._needed["exact_compact_name"].add((country, _compact(name)))
            if address:
                self._needed["exact_address"].add(address)
                self._needed["exact_address_country"].add((address, country))
                self._needed["exact_compact_address"].add((country, _compact(address)))
            for token in _tokens(name):
                self._needed["name_token"].add(token)
                self._needed["country_name_token"].add((country, token))
            significant = sorted(_significant_name_tokens(name))
            for token in significant:
                self._needed["significant_name_token"].add((country, token))
            for left, right in __import__("itertools").combinations(significant[:6], 2):
                self._needed["significant_name_token_pair"].add((country, (left, right)))
            for token in _tokens(address):
                self._needed["address_token"].add(token)
                self._needed["country_address_token"].add((country, token))
            address_tokens = sorted(_significant_address_tokens(address))[:6]
            for number in _address_numbers(address):
                self._needed["address_number"].add((country, number))
                for token in address_tokens:
                    self._needed["address_number_token"].add((country, number, token))
            for prefix in _prefixes(name, prefix_length):
                self._needed["name_prefix"].add(prefix)
            for prefix in _prefixes(address, prefix_length):
                self._needed["address_prefix"].add(prefix)
            for prefix in _compact_prefixes(name, prefix_length + 2):
                self._needed["compact_name_prefix"].add((country, prefix))
            for prefix in _compact_prefixes(address, prefix_length + 2):
                self._needed["compact_address_prefix"].add((country, prefix))
        for name in self._needed:
            setattr(self, name, {})
        self._disabled = {name: set() for name in self._needed}
        self._counts = {name: {} for name in self._needed}

    def _add(self, name, key, label, limit):
        if key not in self._needed[name] or key in self._disabled[name]:
            return
        count = self._counts[name].get(key, 0) + 1
        self._counts[name][key] = count
        if count > limit:
            getattr(self, name).pop(key, None)
            self._disabled[name].add(key)
            return
        getattr(self, name).setdefault(key, []).append(label)

    def add(self, row):
        label = row["entity_id"]
        name, address, country = row["name_norm"], row["address_norm"], row["country_norm"]
        if name:
            self._add("exact_name", name, label, EXACT_POSTING_LIMIT)
            self._add("exact_name_country", (name, country), label, EXACT_POSTING_LIMIT)
            self._add("exact_compact_name", (country, _compact(name)), label, EXACT_POSTING_LIMIT)
        if address:
            self._add("exact_address", address, label, EXACT_POSTING_LIMIT)
            self._add("exact_address_country", (address, country), label, EXACT_POSTING_LIMIT)
            self._add("exact_compact_address", (country, _compact(address)), label, EXACT_POSTING_LIMIT)
        for token in _tokens(name):
            self._add("name_token", token, label, POSTING_LIMIT)
            self._add("country_name_token", (country, token), label, POSTING_LIMIT)
        significant = sorted(_significant_name_tokens(name))
        for token in significant:
            self._add("significant_name_token", (country, token), label, POSTING_LIMIT)
        for pair in __import__("itertools").combinations(significant[:6], 2):
            self._add("significant_name_token_pair", (country, pair), label, POSTING_LIMIT)
        for token in _tokens(address):
            self._add("address_token", token, label, POSTING_LIMIT)
            self._add("country_address_token", (country, token), label, POSTING_LIMIT)
        address_tokens = sorted(_significant_address_tokens(address))[:6]
        for number in _address_numbers(address):
            self._add("address_number", (country, number), label, POSTING_LIMIT)
            for token in address_tokens:
                self._add("address_number_token", (country, number, token), label, POSTING_LIMIT)
        for prefix in _prefixes(name, self.prefix_length):
            self._add("name_prefix", prefix, label, POSTING_LIMIT)
        for prefix in _prefixes(address, self.prefix_length):
            self._add("address_prefix", prefix, label, POSTING_LIMIT)
        for prefix in _compact_prefixes(name, self.prefix_length + 2):
            self._add("compact_name_prefix", (country, prefix), label, POSTING_LIMIT)
        for prefix in _compact_prefixes(address, self.prefix_length + 2):
            self._add("compact_address_prefix", (country, prefix), label, POSTING_LIMIT)


def _stream_source_candidates(path, s1, budget, chunk_size=100_000):
    target_index = _StreamingTargetIndex(s1)
    usecols = ["entity_id", "business_name", "business_address", "country"]
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             encoding="utf-8-sig", usecols=usecols, chunksize=chunk_size):
        for row in _blocking_frame(chunk).itertuples(index=False, name=None):
            target_index.add(dict(zip(chunk.columns.tolist() + ["name_norm", "address_norm", "country_norm"], row)))
    target_stub = pd.DataFrame(index=[])
    return generate_candidates(s1, target_stub, max_candidates=budget, index=target_index, return_provenance=True)


def run_benchmark(data_root, report_dir, sample_size=5000, budgets=(50, 80, 120, 160), random_state=42):
    paths = discover_files(Path(data_root))
    ground_truth = load_tsv(paths["train_ground_truth"])
    selected_gt, _ = _sample_ids(ground_truth, sample_size, random_state)
    selected_ids = selected_gt["source1_entity_id"].tolist()
    s1 = _load_rows_by_id(paths["train_source1"], selected_ids)
    s1 = _blocking_frame(s1).reset_index(drop=True)
    truth = _truth_by_id(selected_gt)
    needed_target_ids = {target_id for ids in truth.values() for target_id in ids}
    target_by_id = {}
    for source_key in ("train_source2", "train_source3"):
        for chunk in pd.read_csv(paths[source_key], sep="\t", dtype=str, keep_default_na=False,
                                 encoding="utf-8-sig", chunksize=100_000):
            matches = chunk[chunk["entity_id"].isin(needed_target_ids)]
            if len(matches):
                target_by_id.update(_blocking_frame(matches).set_index("entity_id").to_dict(orient="index"))
    results = {}
    missed_rows = []

    for budget in budgets:
        cm2, prov2 = _stream_source_candidates(paths["train_source2"], s1, budget)
        cm3, prov3 = _stream_source_candidates(paths["train_source3"], s1, budget)
        candidate_ids = {
            index: set(cm2.get(index, [])) | set(cm3.get(index, []))
            for index in s1.index
        }
        totals = Counter()
        recovered = Counter()
        candidate_counts = []
        singleton_counts = []
        missed = []
        for index, row in s1.iterrows():
            true_ids = set(truth.get(row["entity_id"], []))
            candidates = candidate_ids[index]
            candidate_counts.append(len(candidates))
            if not true_ids:
                singleton_counts.append(len(candidates))
            bucket = _bucket(len(true_ids))
            totals[bucket] += len(true_ids)
            recovered[bucket] += len(true_ids & candidates)
            missed.extend((row["entity_id"], target_id, "s2" if target_id.startswith("S2-") else "s3")
                          for target_id in true_ids - candidates)
        nonzero_total = sum(totals.values())
        s2_total = s2_recovered = s3_total = s3_recovered = 0
        for index, row in s1.iterrows():
            candidate_set = candidate_ids[index]
            for target_id in truth.get(row["entity_id"], []):
                if target_id.startswith("S2-"):
                    s2_total += 1
                    s2_recovered += target_id in candidate_set
                elif target_id.startswith("S3-"):
                    s3_total += 1
                    s3_recovered += target_id in candidate_set
        metrics = {
            "overall_recall": sum(recovered.values()) / nonzero_total if nonzero_total else None,
            "s2_recall": s2_recovered / s2_total if s2_total else None,
            "s3_recall": s3_recovered / s3_total if s3_total else None,
            "true_ids": nonzero_total,
            "recovered_ids": sum(recovered.values()),
            "missed_ids": nonzero_total - sum(recovered.values()),
            "average_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            "median_candidates": float(np.median(candidate_counts)) if candidate_counts else 0.0,
            "p90_candidates": float(np.percentile(candidate_counts, 90)) if candidate_counts else 0.0,
            "p95_candidates": float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0,
            "p99_candidates": float(np.percentile(candidate_counts, 99)) if candidate_counts else 0.0,
            "max_candidates": max(candidate_counts, default=0),
            "singleton": {"s1_rows": len(singleton_counts),
                          "zero_candidate_rows": sum(value == 0 for value in singleton_counts),
                          "zero_candidate_percent": 100 * sum(value == 0 for value in singleton_counts) / len(singleton_counts) if singleton_counts else 0.0,
                          "average_candidates": float(np.mean(singleton_counts)) if singleton_counts else 0.0,
                          "median_candidates": float(np.median(singleton_counts)) if singleton_counts else 0.0},
            "by_match_count": {bucket: {"s1_rows": sum(_bucket(len(truth.get(row["entity_id"], []))) == bucket for _, row in s1.iterrows()),
                                         "true_ids": totals[bucket], "recovered_ids": recovered[bucket],
                                         "recall": recovered[bucket] / totals[bucket] if totals[bucket] else None}
                               for bucket in BUCKETS},
            "missed": missed,
        }
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
        s2_recall = metric.get("s2_recall")
        s3_recall = metric.get("s3_recall")
        lines.append(
            f"| {budget} | {metric['average_candidates']:.2f} | {metric['overall_recall']:.4f} | "
            f"{s2_recall:.4f} | {s3_recall:.4f} |"
            if s2_recall is not None and s3_recall is not None
            else f"| {budget} | {metric['average_candidates']:.2f} | {metric['overall_recall']:.4f} | n/a | n/a |"
        )
    lines.extend(["", "Zero-match entities are excluded from recall and reported under `singleton` in the JSON.", "", "Detailed metrics are in `candidate_recall_benchmark.json`."])
    (report_dir / "candidate_recall_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    return output
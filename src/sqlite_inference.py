import csv
import json
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import (
    EXACT_POSTING_LIMIT, POSTING_LIMIT, _address_numbers, _compact,
    _compact_prefixes, _prefixes, _significant_address_tokens,
    _significant_name_tokens, _tokens, generate_candidates,
)
from .candidate_benchmark import _blocking_frame
from .features import make_feature_matrix
from .normalization import add_normalized

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
INDEX_NAMES = (
    "exact_name", "exact_address", "exact_name_country", "exact_address_country",
    "exact_compact_name", "exact_compact_address", "name_token", "country_name_token",
    "significant_name_token", "significant_name_token_pair", "address_token",
    "country_address_token", "address_number", "address_number_token", "name_prefix",
    "address_prefix", "compact_name_prefix", "compact_address_prefix",
)


def _key(value):
    if isinstance(value, tuple):
        return "\x1f".join(_key(item) for item in value)
    return str(value)


def _keys_for_row(row, prefix_length=4):
    name, address, country = row["name_norm"], row["address_norm"], row["country_norm"]
    out = defaultdict(set)
    if name:
        out["exact_name"].add(name)
        out["exact_name_country"].add((name, country))
        out["exact_compact_name"].add((country, _compact(name)))
    if address:
        out["exact_address"].add(address)
        out["exact_address_country"].add((address, country))
        out["exact_compact_address"].add((country, _compact(address)))
    for token in _tokens(name):
        out["name_token"].add(token); out["country_name_token"].add((country, token))
    significant = sorted(_significant_name_tokens(name))
    for token in significant: out["significant_name_token"].add((country, token))
    for pair in combinations(significant[:6], 2): out["significant_name_token_pair"].add((country, pair))
    for token in _tokens(address):
        out["address_token"].add(token); out["country_address_token"].add((country, token))
    address_tokens = sorted(_significant_address_tokens(address))[:6]
    for number in _address_numbers(address):
        out["address_number"].add((country, number))
        for token in address_tokens: out["address_number_token"].add((country, number, token))
    for prefix in _prefixes(name, prefix_length): out["name_prefix"].add(prefix)
    for prefix in _prefixes(address, prefix_length): out["address_prefix"].add(prefix)
    for prefix in _compact_prefixes(name, prefix_length + 2): out["compact_name_prefix"].add((country, prefix))
    for prefix in _compact_prefixes(address, prefix_length + 2): out["compact_address_prefix"].add((country, prefix))
    return out


def build_index(tsv_path, db_path, chunk_size=100_000, prefix_length=4):
    db_path = Path(db_path)
    if db_path.exists(): return
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE rows (entity_id TEXT PRIMARY KEY, name_norm TEXT, address_norm TEXT, country_norm TEXT, ordinal INTEGER)")
    con.execute("CREATE TABLE postings (index_name TEXT, key TEXT, entity_id TEXT, ordinal INTEGER, PRIMARY KEY(index_name,key,entity_id))")
    con.execute("CREATE INDEX postings_lookup ON postings(index_name,key,ordinal)")
    counts = {}
    disabled = set()
    ordinal = 0
    for chunk in pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, usecols=COLUMNS, chunksize=chunk_size):
        normalized = _blocking_frame(chunk)
        rows = []
        postings = []
        for row in normalized.itertuples(index=False):
            data = row._asdict(); entity_id = data["entity_id"]
            rows.append((entity_id, data["name_norm"], data["address_norm"], data["country_norm"], ordinal))
            for index_name, keys in _keys_for_row(data, prefix_length).items():
                limit = EXACT_POSTING_LIMIT if index_name.startswith("exact") else POSTING_LIMIT
                for raw_key in keys:
                    encoded = (index_name, _key(raw_key))
                    if encoded in disabled: continue
                    count = counts.get(encoded, 0) + 1; counts[encoded] = count
                    if count > limit:
                        con.execute("DELETE FROM postings WHERE index_name=? AND key=?", encoded)
                        disabled.add(encoded); continue
                    postings.append((index_name, encoded[1], entity_id, ordinal))
            ordinal += 1
        con.executemany("INSERT INTO rows VALUES (?,?,?,?,?)", rows)
        con.executemany("INSERT OR IGNORE INTO postings VALUES (?,?,?,?)", postings)
        con.commit()
    con.execute("CREATE INDEX rows_entity ON rows(entity_id)")
    con.commit(); con.close()


class SqliteBatchIndex:
    def __init__(self, con, s1, prefix_length=4):
        self.prefix_length = prefix_length
        self.posting_limit = POSTING_LIMIT
        self._maps = {name: defaultdict(list) for name in INDEX_NAMES}
        needed = {name: set() for name in INDEX_NAMES}
        for _, row in s1[["name_norm","address_norm","country_norm"]].iterrows():
            for name, keys in _keys_for_row(row.to_dict(), prefix_length).items():
                needed[name].update(_key(key) for key in keys)
        for name, keys in needed.items():
            values = list(keys)
            for start in range(0, len(values), 900):
                part = values[start:start+900]
                if not part: continue
                marks = ",".join("?" for _ in part)
                query = f"SELECT key,entity_id FROM postings WHERE index_name=? AND key IN ({marks}) ORDER BY ordinal"
                for key, entity_id in con.execute(query, [name, *part]):
                    self._maps[name][key].append(entity_id)
            setattr(self, name, self._maps[name])


def generate_batch_candidates(s1, con, budget, prefix_length=4, **kwargs):
    index = SqliteBatchIndex(con, s1, prefix_length)
    return generate_candidates(s1, pd.DataFrame(index=[]), max_candidates=budget, index=index, return_provenance=True, **kwargs)


def fetch_rows(con, ids):
    ids = list(ids); rows = []
    for start in range(0, len(ids), 900):
        part = ids[start:start+900]; marks = ",".join("?" for _ in part)
        rows.extend(con.execute(f"SELECT entity_id,name_norm,address_norm,country_norm FROM rows WHERE entity_id IN ({marks})", part))
    return pd.DataFrame(rows, columns=["entity_id","name_norm","address_norm","country_norm"])


def run_sqlite_inference(cfg, model_pack, batch_size=10_000, chunk_size=100_000, sample_s1=None):
    from .data_loader import discover_files
    paths = discover_files(cfg.data_root)
    db_dir = Path(cfg.experiment_dir) / "sqlite_indexes"; db_dir.mkdir(parents=True, exist_ok=True)
    build_index(paths["test_source2"], db_dir / "source2.db", chunk_size, cfg.blocking_prefix_length)
    build_index(paths["test_source3"], db_dir / "source3.db", chunk_size, cfg.blocking_prefix_length)
    con2, con3 = sqlite3.connect(db_dir / "source2.db"), sqlite3.connect(db_dir / "source3.db")
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    candidate_path, matching_path = out / "candidate_pairs.tsv", out / "matching_results.tsv"
    totals = defaultdict(int); first = True
    reader = pd.read_csv(paths["test_source1"], sep="\t", dtype=str, keep_default_na=False,
                         usecols=COLUMNS, chunksize=batch_size, nrows=sample_s1)
    def score(con, s1, cm):
        target_ids = {x for values in cm.values() for x in values}; target = fetch_rows(con, target_ids).drop_duplicates("entity_id").reset_index(drop=True)
        lookup = {value: i for i, value in target.entity_id.items()}; indexed = {i: [lookup[x] for x in ids if x in lookup] for i, ids in cm.items()}
        X, pairs = make_feature_matrix(s1, target, indexed); probs = model_pack["model"].predict_proba(np.ascontiguousarray(X, dtype=np.float32))[:,1] if len(X) else np.array([])
        candidates = {x: [] for x in s1.entity_id}; matches = {x: [] for x in s1.entity_id}
        for (a,b), probability in zip(pairs, probs):
            sid, tid = s1.loc[a,"entity_id"], target.loc[b,"entity_id"]; candidates[sid].append(tid)
            if probability >= float(model_pack["threshold"]): matches[sid].append(tid)
        return candidates, matches, len(X)
    for raw in reader:
        s1 = add_normalized(raw.reset_index(drop=True))
        cm2, _ = generate_batch_candidates(s1, con2, cfg.max_candidates_per_s1, cfg.blocking_prefix_length)
        cm3, _ = generate_batch_candidates(s1, con3, cfg.max_candidates_per_s1, cfg.blocking_prefix_length)
        c2,m2,f2=score(con2,s1,cm2); c3,m3,f3=score(con3,s1,cm3)
        with candidate_path.open("a",encoding="utf-8",newline="") as cf, matching_path.open("a",encoding="utf-8",newline="") as mf:
            cw=csv.DictWriter(cf,fieldnames=["source1_entity_id","candidate_entity_ids"],delimiter="\t"); mw=csv.DictWriter(mf,fieldnames=["source1_entity_id","matched_entity_ids"],delimiter="\t")
            if first: cw.writeheader(); mw.writeheader(); first=False
            for sid in s1.entity_id:
                candidates=sorted(set(c2[sid]+c3[sid])); matches=sorted(set(m2[sid]+m3[sid])); assert set(matches)<=set(candidates)
                cw.writerow({"source1_entity_id":sid,"candidate_entity_ids":",".join(candidates)}); mw.writerow({"source1_entity_id":sid,"matched_entity_ids":",".join(matches)})
                totals.update(source1_rows=1,candidate_rows=len(candidates),matched_rows=len(matches),zero_match_s1=not matches,multi_match_s1=len(matches)>1,feature_rows=f2+f3)
    con2.close(); con3.close(); (Path(cfg.report_dir)/"streaming_inference.json").write_text(json.dumps(dict(totals),indent=2),encoding="utf-8"); return dict(totals)

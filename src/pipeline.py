import argparse, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd

from .config import Config
from .data_loader import load_all
from .normalization import add_normalized
from .blocking import build_target_index, generate_candidates
from .features import make_feature_matrix, FEATURE_NAMES
from .ground_truth import parse_ground_truth
from .evaluate import macro_f05, tune_threshold
from .eda import run_eda
from .candidate_benchmark import run_benchmark
from .validate_outputs import validate_outputs
from .streaming_inference import run_streaming_inference
from .sqlite_inference import run_sqlite_inference

class _BoosterClassifier:
    def __init__(self, booster=None, model_text=None):
        if model_text is None:
            if booster is None:
                raise ValueError("booster or model_text is required")
            model_text = booster.model_to_string()
        self.model_text = model_text
        self._booster = None
        self._model_path = None

    @property
    def booster(self):
        if not hasattr(self, "model_text") and "booster" in self.__dict__:
            legacy_booster = self.__dict__.pop("booster")
            self.model_text = legacy_booster.model_to_string()
            self._booster = None
        if self._booster is None:
            import lightgbm as lgb
            import atexit
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as handle:
                handle.write(self.model_text.encode("utf-8"))
                model_path = handle.name
            self._model_path = model_path
            def cleanup_model_file(path=model_path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            atexit.register(cleanup_model_file)
            self._booster = lgb.Booster(model_file=model_path)
        return self._booster

    def predict_proba(self, features):
        probabilities = np.asarray(self.booster.predict(features), dtype=np.float64)
        return np.column_stack((1.0 - probabilities, probabilities))


def prepare(dfs):
    return {k:add_normalized(v) if k.startswith(("train_source","test_source")) else v for k,v in dfs.items()}

def candidate_recall(candidate_map, truth, target):
    id_by_index = target["entity_id"].to_dict()
    hit = total = 0
    for s1_index, ids in truth.items():
        candidate_ids = {id_by_index[i] for i in candidate_map.get(s1_index, [])}
        total += len(ids)
        hit += len(set(ids) & candidate_ids)
    return hit/total if total else 1.0


def evaluate_candidate_recall(cfg, dfs):
    s1, s2, s3 = dfs["train_source1"], dfs["train_source2"], dfs["train_source3"]
    truth = parse_ground_truth(dfs["train_ground_truth"])
    s2_index = build_target_index(s2, prefix_length=cfg.blocking_prefix_length)
    s3_index = build_target_index(s3, prefix_length=cfg.blocking_prefix_length)
    cm2 = generate_indexed_candidates(s1, s2, cfg, s2_index)
    cm3 = generate_indexed_candidates(s1, s3, cfg, s3_index)
    counts = [len(cm2.get(i, [])) + len(cm3.get(i, [])) for i in s1.index]
    hits = total = 0
    by_count = {}
    for index, row in s1.iterrows():
        true_ids = set(truth.get(row["entity_id"], []))
        candidate_ids = {s2.loc[i, "entity_id"] for i in cm2.get(index, [])}
        candidate_ids |= {s3.loc[i, "entity_id"] for i in cm3.get(index, [])}
        found = len(true_ids & candidate_ids)
        hits += found
        total += len(true_ids)
        bucket = by_count.setdefault(str(len(true_ids)), [0, 0])
        bucket[0] += found
        bucket[1] += len(true_ids)
    report = {
        "rows": len(s1),
        "matched_ids": total,
        "matched_ids_found": hits,
        "candidate_recall": hits / total if total else 1.0,
        "average_candidates": float(np.mean(counts)) if counts else 0.0,
        "median_candidates": float(np.median(counts)) if counts else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if counts else 0.0,
        "max_candidates": max(counts, default=0),
        "by_true_match_count": {
            key: {"found": value[0], "total": value[1], "recall": value[0] / value[1] if value[1] else 1.0}
            for key, value in by_count.items()
        },
    }
    (cfg.report_dir / "candidate_recall.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

def build_pairs(s1, s2, s3, candidate_maps):
    rows=[]
    for target_name,target in (("s2",s2),("s3",s3)):
        for s1i,tids in candidate_maps[target_name].items():
            for ti in tids:
                rows.append((s1i, target_name, ti))
    return rows


def generate_indexed_candidates(s1, target, cfg, target_index=None):
    target_index = target_index or build_target_index(
        target, prefix_length=cfg.blocking_prefix_length
    )
    return generate_candidates(
        s1,
        target,
        max_candidates=cfg.max_candidates_per_s1,
        index=target_index,
        exact_limit=cfg.exact_candidate_limit,
        token_limit=cfg.token_candidate_limit,
        prefix_limit=cfg.prefix_candidate_limit,
        max_name_tokens=cfg.max_name_block_tokens,
        max_address_tokens=cfg.max_address_block_tokens,
        exact_family_fraction=cfg.exact_family_fraction,
        name_family_fraction=cfg.name_family_fraction,
        address_family_fraction=cfg.address_family_fraction,
        number_family_fraction=cfg.number_family_fraction,
    )

def truth_sets(gt, s2, s3):
    id_to_idx={}
    for name,df in (("s2",s2),("s3",s3)):
        for idx,eid in df["entity_id"].items():
            id_to_idx[eid]=(name,idx)
    out={}
    for s1eid, ids in gt.items():
        out[s1eid]=ids
    return out, id_to_idx

def train(cfg, dfs):
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    s1,s2,s3 = dfs["train_source1"],dfs["train_source2"],dfs["train_source3"]
    gt = parse_ground_truth(dfs["train_ground_truth"])

    s1_ids=s1["entity_id"].tolist()
    tr_ids, va_ids=train_test_split(s1_ids,test_size=cfg.validation_fraction,random_state=cfg.random_state)
    tr_set=set(tr_ids); va_set=set(va_ids)
    s1tr=s1[s1["entity_id"].isin(tr_set)].copy()
    s1va=s1[s1["entity_id"].isin(va_set)].copy()

    print("Generating training candidates...")
    s2_index = build_target_index(s2, prefix_length=cfg.blocking_prefix_length)
    s3_index = build_target_index(s3, prefix_length=cfg.blocking_prefix_length)
    cm2tr=generate_indexed_candidates(s1tr,s2,cfg,s2_index)
    cm3tr=generate_indexed_candidates(s1tr,s3,cfg,s3_index)
    cm2va=generate_indexed_candidates(s1va,s2,cfg,s2_index)
    cm3va=generate_indexed_candidates(s1va,s3,cfg,s3_index)

    # Map local dataframe indices to IDs for labels.
    id2s2={eid:i for i,eid in s2["entity_id"].items()}
    id2s3={eid:i for i,eid in s3["entity_id"].items()}

    def make_training(cm2,cm3,s1df):
        cm={"s2":cm2,"s3":cm3}
        X,pairs=make_feature_matrix(s1df,s2,cm2)
        X3,pairs3=make_feature_matrix(s1df,s3,cm3)
        X=np.vstack([X,X3]); meta=[("s2",a,b) for a,b in pairs]+[("s3",a,b) for a,b in pairs3]
        y=[]
        for src,a,b in meta:
            eid=s1df.loc[a,"entity_id"]
            tid=(s2 if src=="s2" else s3).loc[b,"entity_id"]
            y.append(int(tid in set(gt.get(eid,[]))))
        return X,np.asarray(y),meta

    X,y,meta=make_training(cm2tr,cm3tr,s1tr)
    pos=np.where(y==1)[0]
    neg=np.where(y==0)[0]
    rng=np.random.default_rng(cfg.random_state)
    negative_limit = min(len(neg), max(len(pos) * cfg.train_negative_multiplier, 1))
    hard_negative_count = min(len(neg), negative_limit // 2)
    if hard_negative_count:
        hard_score = X[neg, 17] + X[neg, 4] + X[neg, 10]
        hard_order = np.lexsort((neg, -hard_score))
        hard_neg = neg[hard_order[:hard_negative_count]]
        remaining_neg = np.setdiff1d(neg, hard_neg, assume_unique=False)
    else:
        hard_neg = np.array([], dtype=int)
        remaining_neg = neg
    easy_count = min(len(remaining_neg), negative_limit - len(hard_neg))
    easy_neg = rng.choice(remaining_neg, size=easy_count, replace=False) if easy_count else np.array([], dtype=int)
    keep_neg = np.concatenate([hard_neg, easy_neg])
    keep=np.concatenate([pos,keep_neg])
    Xtr=np.ascontiguousarray(X[keep], dtype=np.float32)
    ytr=np.ascontiguousarray(y[keep], dtype=np.int32)
    if len(np.unique(ytr)) < 2 or len(ytr) < 2:
        raise ValueError(
            "Training sample has fewer than two classes. Increase --sample-s1 "
            "or use the full training data; the bounded prefix sample may contain "
            "only no-match entities."
        )

    model_kwargs = {
        "n_estimators": 700, "learning_rate": 0.035, "num_leaves": 31,
        "max_depth": -1,
        "reg_lambda": 2.0, "random_state": cfg.random_state,
        "objective": "binary", "device_type": cfg.device_type,
        "n_jobs": 1,
        "force_col_wise": True, "max_bin": 63, "min_data_in_bin": 1,
        "min_data_in_leaf": 5, "verbosity": -1,
    }
    if cfg.device_type == "cuda":
        model_kwargs.update({"gpu_device_id": cfg.gpu_device_id, "subsample": 0.85, "colsample_bytree": 0.90})
    model=LGBMClassifier(**model_kwargs)
    try:
        model.fit(Xtr,ytr,feature_name=FEATURE_NAMES)
    except OSError as exc:
        if cfg.device_type != "cpu":
            raise
        booster = lgb.train(
            {
                "objective": "binary",
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": -1,
                "lambda_l2": 2.0,
                "verbosity": -1,
                "force_col_wise": True,
                "max_bin": 63,
                "min_data_in_bin": 1,
                "min_data_in_leaf": 5,
                "num_threads": 1,
                "seed": cfg.random_state,
            },
            lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES),
            num_boost_round=700,
        )
        model = _BoosterClassifier(booster)
    except Exception as exc:
        if cfg.device_type == "cuda" and "CUDA Tree Learner was not enabled" in str(exc):
            raise RuntimeError(
                "CUDA was requested, but the installed LightGBM build has no CUDA support. "
                "Install a CUDA-enabled LightGBM build or rerun with --device cpu."
            ) from exc
        raise

    if hasattr(model, "booster_"):
        model = _BoosterClassifier(model.booster_)

    # Validation probabilities.
    Xv,pairs_v=make_feature_matrix(s1va,s2,cm2va)
    Xv3,pairs_v3=make_feature_matrix(s1va,s3,cm3va)
    pv=model.predict_proba(Xv)[:,1] if len(Xv) else np.array([])
    pv3=model.predict_proba(Xv3)[:,1] if len(Xv3) else np.array([])

    prob_pairs=[]
    for (a,b),p in zip(pairs_v,pv):
        prob_pairs.append((s1va.loc[a,"entity_id"],s2.loc[b,"entity_id"],float(p)))
    for (a,b),p in zip(pairs_v3,pv3):
        prob_pairs.append((s1va.loc[a,"entity_id"],s3.loc[b,"entity_id"],float(p)))

    truth_va={eid:gt.get(eid,[]) for eid in s1va["entity_id"]}
    thresholds=np.arange(cfg.threshold_low,cfg.threshold_high+1e-9,cfg.threshold_step)
    best=tune_threshold(prob_pairs,truth_va,thresholds)

    # Candidate recall on validation.
    cand_lookup={}
    total_true=0; hit_true=0
    for s1i,tids in cm2va.items():
        eid=s1va.loc[s1i,"entity_id"]
        cand_lookup.setdefault(eid,set()).update(s2.loc[tids,"entity_id"].tolist())
    for s1i,tids in cm3va.items():
        eid=s1va.loc[s1i,"entity_id"]
        cand_lookup.setdefault(eid,set()).update(s3.loc[tids,"entity_id"].tolist())
    for eid,trueids in truth_va.items():
        total_true += len(trueids)
        hit_true += len(set(trueids)&cand_lookup.get(eid,set()))
    best["candidate_recall"]=hit_true/total_true if total_true else 1.0
    best["avg_candidates"]=float(np.mean([len(cm2va.get(i,[]))+len(cm3va.get(i,[])) for i in cm2va])) if cm2va else 0.0
    best["training_positive_count"] = int(len(pos))
    best["training_easy_negative_count"] = int(len(easy_neg))
    best["training_hard_negative_count"] = int(len(hard_neg))
    best["training_pair_count"] = int(len(ytr))

    cfg.model_dir.mkdir(exist_ok=True)
    joblib.dump({"model":model,"feature_names":FEATURE_NAMES,"threshold":best["threshold"]},cfg.model_dir/"matcher.joblib")
    (cfg.report_dir/"validation.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
    print(json.dumps(best,indent=2))
    return best

def inference(cfg, dfs):
    pack=joblib.load(cfg.model_dir/"matcher.joblib")
    model=pack["model"]; threshold=float(pack["threshold"])
    s1,s2,s3=dfs["test_source1"],dfs["test_source2"],dfs["test_source3"]
    s2_index = build_target_index(s2, prefix_length=cfg.blocking_prefix_length)
    s3_index = build_target_index(s3, prefix_length=cfg.blocking_prefix_length)
    cm2=generate_indexed_candidates(s1,s2,cfg,s2_index)
    cm3=generate_indexed_candidates(s1,s3,cfg,s3_index)

    X2,p2=make_feature_matrix(s1,s2,cm2)
    X3,p3=make_feature_matrix(s1,s3,cm3)
    prob2=model.predict_proba(X2)[:,1] if len(X2) else np.array([])
    prob3=model.predict_proba(X3)[:,1] if len(X3) else np.array([])

    candidates={eid:[] for eid in s1["entity_id"]}
    matches={eid:[] for eid in s1["entity_id"]}

    for (a,b),p in zip(p2,prob2):
        eid=s1.loc[a,"entity_id"]; tid=s2.loc[b,"entity_id"]
        candidates[eid].append(tid)
        if p >= threshold: matches[eid].append(tid)
    for (a,b),p in zip(p3,prob3):
        eid=s1.loc[a,"entity_id"]; tid=s3.loc[b,"entity_id"]
        candidates[eid].append(tid)
        if p >= threshold: matches[eid].append(tid)

    def clean(xs): return sorted(set(xs))
    matching_rows=[{"source1_entity_id":eid,"matched_entity_ids":",".join(clean(matches[eid]))} for eid in s1["entity_id"]]
    candidate_rows=[{"source1_entity_id":eid,"candidate_entity_ids":",".join(clean(candidates[eid]))} for eid in s1["entity_id"]]
    pd.DataFrame(matching_rows).to_csv(cfg.output_dir/"matching_results.tsv",sep="\t",index=False)
    pd.DataFrame(candidate_rows).to_csv(cfg.output_dir/"candidate_pairs.tsv",sep="\t",index=False)
    print("Wrote:",cfg.output_dir/"matching_results.tsv")
    print("Wrote:",cfg.output_dir/"candidate_pairs.tsv")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default=r"D:\rig\mlchallenge\student_resource")
    ap.add_argument("--mode",choices=["eda","candidate-recall","benchmark","train","inference","validate","all"],default="all")
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--sample-size", type=int, default=5000)
    ap.add_argument("--candidate-budget", type=int, action="append", default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="LightGBM training device; CUDA is the default."
    )
    ap.add_argument("--gpu-device-id", type=int, default=0)
    ap.add_argument("--stream-batch-size", type=int, default=50_000)
    ap.add_argument("--stream-chunk-size", type=int, default=100_000)
    ap.add_argument("--sqlite-inference", action="store_true")
    args=ap.parse_args()
    cfg=Config(Path(args.data_root)); cfg.prepare()
    cfg.random_state = args.random_state
    cfg.device_type = args.device
    cfg.gpu_device_id = args.gpu_device_id
    if args.mode == "benchmark":
        budgets = tuple(args.candidate_budget or (50, 80, 120, 160))
        run_benchmark(cfg.data_root, cfg.report_dir, args.sample_size, budgets, args.random_state)
        return
    if args.mode == "validate":
        print(json.dumps(validate_outputs(cfg.data_root, cfg.output_dir), indent=2))
        return
    if args.mode == "inference":
        pack = joblib.load(cfg.model_dir / "matcher.joblib")
        if args.sqlite_inference:
            print(json.dumps(run_sqlite_inference(
                cfg, pack, batch_size=args.stream_batch_size, chunk_size=args.stream_chunk_size,
                sample_s1=args.sample_s1
            ), indent=2))
            return
        print(json.dumps(run_streaming_inference(
            cfg, pack, batch_size=args.stream_batch_size, chunk_size=args.stream_chunk_size,
            sample_s1=args.sample_s1
        ), indent=2))
        return
    split = "train" if args.mode in {"eda", "candidate-recall", "train"} else "test"
    paths,dfs=load_all(cfg.data_root, nrows=args.sample_s1, split=split)
    print("Discovered files:")
    for k,p in paths.items(): print(f"  {k}: {p}")
    dfs=prepare(dfs)
    if args.mode in ("eda","all"):
        print(run_eda(dfs,cfg.report_dir/"eda_report.md"))
    if args.mode == "candidate-recall":
        evaluate_candidate_recall(cfg,dfs)
    if args.mode in ("train","all"):
        if args.mode == "all":
            _, train_dfs = load_all(cfg.data_root, nrows=args.sample_s1, split="train")
            train_dfs = prepare(train_dfs)
        else:
            train_dfs = dfs
        train(cfg,train_dfs)
    if args.mode in ("inference","all"):
        if args.mode == "all":
            _, test_dfs = load_all(cfg.data_root, nrows=args.sample_s1, split="test")
            test_dfs = prepare(test_dfs)
        else:
            test_dfs = dfs
        inference(cfg,test_dfs)

if __name__=="__main__":
    main()

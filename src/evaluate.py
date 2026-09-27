import numpy as np

def fbeta(precision, recall, beta=0.5):
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1+b2)*precision*recall/(b2*precision+recall)

def macro_f05(pred_map, truth_map):
    scores=[]; tp=fp=fn=0; singleton_correct=0
    for s1, truth in truth_map.items():
        truth=set(truth)
        pred=set(pred_map.get(s1, []))
        t=len(pred & truth); f=len(pred-truth); n=len(truth-pred)
        p=t/(t+f) if t+f else (1.0 if not pred else 0.0)
        r=t/(t+n) if t+n else (1.0 if not pred else 0.0)
        scores.append(fbeta(p,r,.5))
        tp+=t; fp+=f; fn+=n
        if not truth and not pred:
            singleton_correct += 1
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return {
        "f0_5": float(np.mean(scores)) if scores else 0.0,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "singleton_correct": singleton_correct,
        "singleton_total": sum(not bool(v) for v in truth_map.values())
    }

def tune_threshold(prob_pairs, truth_map, thresholds):
    best=None
    for t in thresholds:
        pred={}
        for s1, tid, p in prob_pairs:
            if p >= t:
                pred.setdefault(s1, []).append(tid)
        m=macro_f05(pred, truth_map)
        m["threshold"]=float(t)
        if best is None or m["f0_5"] > best["f0_5"]:
            best=m
    return best

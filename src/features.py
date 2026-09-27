import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_ratio","name_partial","name_token_set","name_token_sort",
    "name_jaccard","name_len_ratio",
    "addr_ratio","addr_partial","addr_token_set","addr_token_sort",
    "addr_jaccard","addr_len_ratio",
    "address_missing_a","address_missing_b","country_match","name_exact","address_exact",
    "combined_ratio","both_strong","name_addr_interaction"
]

def jaccard(a,b):
    A, B = set(str(a).split()), set(str(b).split())
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A&B)/len(A|B)

def len_ratio(a,b):
    la, lb = len(str(a)), len(str(b))
    return min(la,lb)/max(la,lb) if max(la,lb) else 1.0

def pair_features(a,b):
    nr = fuzz.ratio(a["name_norm"], b["name_norm"]) / 100.0
    npart = fuzz.partial_ratio(a["name_norm"], b["name_norm"]) / 100.0
    nts = fuzz.token_set_ratio(a["name_norm"], b["name_norm"]) / 100.0
    nsort = fuzz.token_sort_ratio(a["name_norm"], b["name_norm"]) / 100.0
    nj = jaccard(a["name_norm"], b["name_norm"])
    nl = len_ratio(a["name_norm"], b["name_norm"])

    address_a = a["address_norm"]
    address_b = b["address_norm"]
    address_missing_a = float(not address_a)
    address_missing_b = float(not address_b)
    ar = fuzz.ratio(address_a, address_b) / 100.0 if address_a and address_b else 0.0
    apart = fuzz.partial_ratio(address_a, address_b) / 100.0 if address_a and address_b else 0.0
    ats = fuzz.token_set_ratio(address_a, address_b) / 100.0 if address_a and address_b else 0.0
    asort = fuzz.token_sort_ratio(address_a, address_b) / 100.0 if address_a and address_b else 0.0
    aj = jaccard(address_a, address_b) if address_a and address_b else 0.0
    al = len_ratio(address_a, address_b) if address_a and address_b else 0.0

    cm = float(a["country_norm"] == b["country_norm"])
    ne = float(a["name_norm"] == b["name_norm"] and bool(a["name_norm"]))
    ae = float(a["address_norm"] == b["address_norm"] and bool(a["address_norm"]))
    cr = (nr + ar) / 2.0
    bs = float(nr >= .90 and ar >= .75)
    interaction = nr * ar

    return [
        nr,npart,nts,nsort,nj,nl,
        ar,apart,ats,asort,aj,al,address_missing_a,address_missing_b,
        cm,ne,ae,cr,bs,interaction
    ]

def make_feature_matrix(s1, target, candidate_map):
    rows=[]; pairs=[]
    for s1i, tids in candidate_map.items():
        a = s1.loc[s1i]
        for ti in tids:
            b = target.loc[ti]
            rows.append(pair_features(a,b))
            pairs.append((s1i,ti))
    return np.asarray(rows, dtype=np.float32), pairs

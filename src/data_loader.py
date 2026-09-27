from pathlib import Path
import pandas as pd

EXPECTED = {
    "train_source1.tsv": "train_source1",
    "train_source2.tsv": "train_source2",
    "train_source3.tsv": "train_source3",
    "train_ground_truth.tsv": "train_ground_truth",
    "test_source1.tsv": "test_source1",
    "test_source2.tsv": "test_source2",
    "test_source3.tsv": "test_source3",
}

def discover_files(data_root: Path):
    data_root = Path(data_root)
    found = {}
    for p in data_root.rglob("*.tsv"):
        if p.name in EXPECTED and EXPECTED[p.name] not in found:
            found[EXPECTED[p.name]] = p
    return found

def load_tsv(path: Path, nrows=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)

def load_all(data_root: Path, nrows=None):
    paths = discover_files(data_root)
    missing_train = [x for x in ("train_source1","train_source2","train_source3","train_ground_truth") if x not in paths]
    missing_test = [x for x in ("test_source1","test_source2","test_source3") if x not in paths]
    if missing_train or missing_test:
        raise FileNotFoundError(
            f"Could not discover required TSVs under {data_root}. "
            f"Missing train={missing_train}, test={missing_test}. "
            f"Discovered={list(paths)}"
        )
    dfs = {k: load_tsv(v, nrows=nrows) for k,v in paths.items()}
    return paths, dfs

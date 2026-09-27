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
    kwargs = dict(
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )
    if nrows is not None:
        kwargs["nrows"] = nrows
    return pd.read_csv(path, **kwargs)


def load_all(data_root: Path, nrows=None, split="all"):
    if split not in {"all", "train", "test"}:
        raise ValueError("split must be 'all', 'train', or 'test'")
    paths = discover_files(data_root)
    train_keys = ("train_source1", "train_source2", "train_source3", "train_ground_truth")
    test_keys = ("test_source1", "test_source2", "test_source3")
    required_train = split in {"all", "train"}
    required_test = split in {"all", "test"}
    missing_train = [x for x in train_keys if required_train and x not in paths]
    missing_test = [x for x in test_keys if required_test and x not in paths]
    if missing_train or missing_test:
        raise FileNotFoundError(
            f"Could not discover required TSVs under {data_root}. "
            f"Missing train={missing_train}, test={missing_test}. "
            f"Discovered={list(paths)}"
        )
    selected_keys = train_keys if split == "train" else test_keys if split == "test" else train_keys + test_keys
    dfs = {k: load_tsv(paths[k], nrows=nrows) for k in selected_keys}
    return paths, dfs

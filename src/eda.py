from pathlib import Path
import pandas as pd

def run_eda(dfs, report_path: Path):
    lines=["# Amazon ML Challenge — EDA\n"]
    for key,df in dfs.items():
        lines += [
            f"## {key}",
            f"- rows: {len(df):,}",
            f"- columns: {list(df.columns)}",
            f"- duplicate rows: {int(df.duplicated().sum()):,}",
        ]
        for c in df.columns:
            if c in ("entity_id",):
                continue
            miss=int(df[c].eq("").sum()) if df[c].dtype == object else int(df[c].isna().sum())
            lines.append(f"- `{c}` empty/missing: {miss:,}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)

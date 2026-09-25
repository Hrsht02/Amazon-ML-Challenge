"""
Loading and light validation of the challenge's TSV files.

Every file is read with sep="\t", dtype=str, keep_default_na=False so that:
  - empty strings stay empty strings, not NaN (avoids float-NaN bugs
    downstream in string feature code)
  - CRLF line endings (present in the provided files) are handled
    transparently by pandas' C/python parser
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict


REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    for c in REQUIRED_COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    if df.entity_id.duplicated().any():
        dups = df.entity_id[df.entity_id.duplicated()].tolist()
        raise ValueError(f"{path} has duplicate entity_id values: {dups[:5]}...")
    df = df.set_index("entity_id", drop=False)
    return df


def load_split(dataset_dir: Path, split: str) -> Dict[str, pd.DataFrame]:
    """split is 'train' or 'test'. Returns dict with keys s1, s2, s3 (+ gt for train)."""
    d = dataset_dir / split
    out = {
        "s1": _read_source(d / f"{split}_source1.tsv"),
        "s2": _read_source(d / f"{split}_source2.tsv"),
        "s3": _read_source(d / f"{split}_source3.tsv"),
    }
    if split == "train":
        gt_path = d / "train_ground_truth.tsv"
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
        gt["match_list"] = gt["matched_entity_ids"].apply(
            lambda x: [i.strip() for i in x.split(",") if i.strip()]
        )
        out["gt"] = gt.set_index("source1_entity_id", drop=False)
    return out


def s2s3_lookup(s2: pd.DataFrame, s3: pd.DataFrame) -> Dict[str, pd.Series]:
    """Single dict keyed by entity_id across both S2 and S3, for O(1) row lookup."""
    lut = {}
    for df in (s2, s3):
        for eid, row in df.iterrows():
            lut[eid] = row
    return lut

"""
Thin preprocessing layer: fits the data-driven Normalizer (legal-suffix
mining, see normalization.py) on the training business names, and offers
a couple of quick audit helpers used by the training script's console
report (country distribution, non-Latin-script rate, singleton rate) so
every run prints a fresh data-driven sanity check instead of trusting
stale assumptions about the dataset.
"""
from __future__ import annotations
import pandas as pd
from .normalization import Normalizer


def fit_normalizer(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Normalizer:
    names = pd.concat([s1_df.business_name, s2_df.business_name, s3_df.business_name])
    return Normalizer.fit(names)


def quick_audit(s1_df, s2_df, s3_df, gt_df=None) -> dict:
    import re
    def non_latin_rate(df):
        return df.business_name.apply(lambda s: bool(re.search(r"[^\x00-\x7F]", s or ""))).mean()

    report = {
        "n_s1": len(s1_df), "n_s2": len(s2_df), "n_s3": len(s3_df),
        "country_dist_s1": s1_df.country.value_counts().to_dict(),
        "country_dist_s2": s2_df.country.value_counts().to_dict(),
        "country_dist_s3": s3_df.country.value_counts().to_dict(),
        "non_latin_name_rate_s1": round(non_latin_rate(s1_df), 4),
        "non_latin_name_rate_s2": round(non_latin_rate(s2_df), 4),
        "non_latin_name_rate_s3": round(non_latin_rate(s3_df), 4),
    }
    if gt_df is not None:
        n_matches = gt_df.match_list.apply(len)
        report["singleton_rate"] = float((n_matches == 0).mean())
        report["avg_matches_per_s1"] = float(n_matches.mean())
        report["match_count_distribution"] = n_matches.value_counts().sort_index().to_dict()
    return report

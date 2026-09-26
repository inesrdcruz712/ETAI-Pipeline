import pandas as pd

def test_missingness_mechanism(df: pd.DataFrame, target_col: str, candidate_predictors: list) -> dict:
    """
    A first, informal look at *why* values are missing in each column
    that has any missingness -- not a formal statistical test, just
    group comparisons that are enough to flag "this doesn't look
    completely random":

      - is the missing rate different depending on the target? (rows
        missing this column vs. rows that aren't -- do they have a
        different two_year_recid rate?)
      - is the missing rate different across groups of each candidate
        predictor? (e.g. does 'sex' missingness differ by age_cat?)

    A big gap in either suggests MAR/MNAR rather than MCAR, which
    matters for how the column should eventually be imputed.
    """
    results = {}
    missing_cols = [c for c in df.columns if df[c].isna().any()]

    for col in missing_cols:
        is_missing = df[col].isna()
        col_result = {
            "n_missing": int(is_missing.sum()),
            "pct_missing": round(is_missing.mean() * 100, 1),
        }

        if target_col in df.columns and target_col != col:
            col_result["target_rate_when_missing"] = round(df.loc[is_missing, target_col].mean(), 3)
            col_result["target_rate_when_present"] = round(df.loc[~is_missing, target_col].mean(), 3)

        predictor_gaps = {}
        for pred in candidate_predictors:
            if pred not in df.columns or pred == col:
                continue
            miss_rate_by_group = df.groupby(df[pred].astype(str))[col].apply(lambda s: s.isna().mean())
            if len(miss_rate_by_group) > 1:
                predictor_gaps[pred] = round(float(miss_rate_by_group.max() - miss_rate_by_group.min()), 3)
        col_result["missing_rate_spread_by_predictor"] = predictor_gaps

        results[col] = col_result

    return results

def flag_invalid_values(df: pd.DataFrame, rules: dict) -> dict:
    """
    For each column in `rules` (config.yaml -> diagnostics.validity_rules),
    coerces to numeric and flags values outside the declared [min, max].
    Non-numeric entries (e.g. placeholder tokens) are reported separately,
    since they are a different problem from an out-of-range number.
    """
    results = {}
    for col, bounds in rules.items():
        if col not in df.columns:
            continue

        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        non_numeric_mask = raw.notna() & numeric.isna()

        lo = bounds.get("min", float("-inf"))
        hi = bounds.get("max", float("inf"))
        out_of_range_mask = numeric.notna() & ((numeric < lo) | (numeric > hi))

        results[col] = {
            "rule": bounds,
            "n_non_numeric": int(non_numeric_mask.sum()),
            "non_numeric_examples": sorted(set(raw[non_numeric_mask].astype(str)))[:10],
            "n_out_of_range": int(out_of_range_mask.sum()),
            "out_of_range_examples": sorted(numeric[out_of_range_mask].unique().tolist())[:10],
        }
    return results

def find_duplicates(df: pd.DataFrame, id_column: str) -> dict:
    """
    Looks for duplicate records in two ways:
      - exact duplicates: every column identical, including `id_column`
      - near-duplicates: every column identical EXCEPT `id_column`
        (same case data, filed twice under a different id)
    """
    exact_dup_mask = df.duplicated(keep=False)
    other_cols = [c for c in df.columns if c != id_column]
    near_dup_mask = df.duplicated(subset=other_cols, keep=False)

    return {
        "n_exact_duplicates": int(df.duplicated().sum()),
        "n_exact_duplicate_rows_involved": int(exact_dup_mask.sum()),
        "n_near_duplicates": int(df.duplicated(subset=other_cols).sum()),
        "n_near_duplicate_rows_involved": int(near_dup_mask.sum()),
        "example_ids_exact": df.loc[exact_dup_mask, id_column].head(5).tolist(),
        "example_ids_near": df.loc[near_dup_mask, id_column].head(5).tolist(),
    }
"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_diagnostics import find_duplicates, flag_invalid_values, test_missingness_mechanism

def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # only drop rows with missing values in columns the model actually uses.
    # sensitive_attr (race) and the columns in drop_columns (id, decile_score,
    # score_text) are kept aside only for auditing/comparison after training --
    # a missing value there shouldn't cost us a row of otherwise-complete
    # training data. (Their own NaNs, if any, are left as-is here: they never
    # reach the model, and the fairness report / comparison step can decide
    # how to handle them.)
    excluded_cols = [sensitive_attr] + [c for c in drop_columns if c in df.columns]
    feature_and_target_cols = [c for c in df.columns if c not in excluded_cols]
    n_before = len(df)
    df = df.dropna(subset=feature_and_target_cols)
    print(f"[preprocess] dropped {n_before - len(df)} row(s) with missing model-feature values "
          f"(ignoring {excluded_cols}); shape: {df.shape}")

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test

def data_cleaning(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    All data-cleaning steps live in here. Each step is only added once
    we've decided (from diagnose()'s report) how it should be resolved --
    so far that's problems 2 and 3 from the report; more steps get added
    below as we work through the rest.

    Returns a NEW DataFrame; `df` itself is not modified.
    """
    diag_cfg = config["diagnostics"]
    df = df.copy()

    print("[data_cleaning] starting from shape:", df.shape)

    # --- step 1 (problem 2): placeholder tokens -> real NaN --------------
    tokens = set(diag_cfg.get("placeholder_tokens", []))
    tokens.discard("")  # an actual empty cell already reads as NaN on load; nothing to replace there

    print(f"[data_cleaning] step 1/7 - placeholder tokens: looking for {sorted(tokens)}")

    total_replaced = 0
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "str":
            not_null = df[col].notna()
            is_token = df[col][not_null].astype(str).str.strip().isin(tokens)
            token_index = df[col][not_null][is_token].index
            n = len(token_index)
            if n > 0:
                print(f"[data_cleaning]   {col}: {n} placeholder value(s) -> NaN")
                total_replaced += n
            df.loc[token_index, col] = pd.NA
    print(f"[data_cleaning]   total values replaced: {total_replaced}")

    # numeric_text_columns (priors_count, prior_offenses) only load as text
    # because of the placeholder tokens above -- now that those are gone,
    # convert them to numeric.
    for col in diag_cfg.get("numeric_text_columns", []):
        if col in df.columns:
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"[data_cleaning]   {col}: dtype {before_dtype} -> {df[col].dtype}")

    # --- step 2 (problem 3): inconsistent category spelling/casing -------
    canonical = diag_cfg.get("canonical_categories", {})
    print(f"[data_cleaning] step 2/7 - standardizing categories: {list(canonical.keys())}")

    for col, mapping in canonical.items():
        if col not in df.columns:
            print(f"[data_cleaning]   {col}: not in df, skipping")
            continue
        not_null = df[col].notna()
        normalized = df[col][not_null].astype(str).str.strip().str.lower()
        mapped = normalized.map(mapping)

        n_changed = int((mapped.notna() & (mapped != df[col][not_null])).sum())
        n_unmapped = int(mapped.isna().sum())
        print(f"[data_cleaning]   {col}: {n_changed} value(s) rewritten to canonical spelling, "
              f"{n_unmapped} value(s) left untouched (no matching mapping)")
        if n_unmapped > 0:
            leftovers = sorted(set(df[col][not_null][mapped.isna()].astype(str)))
            print(f"[data_cleaning]   {col}: unmapped values -> {leftovers}")

        df.loc[not_null, col] = mapped.where(mapped.notna(), df[col][not_null])

    # --- step 3 (problem 6): drop redundant columns ----------------------
    redundant_cols = [c for c in diag_cfg.get("redundant_columns", []) if c in df.columns]
    print(f"[data_cleaning] step 3/7 - dropping redundant columns: {redundant_cols}")
    df = df.drop(columns=redundant_cols)
    print(f"[data_cleaning]   shape after drop: {df.shape}")

    # --- step 4 (problem 5): drop EXACT duplicate rows only ---------------
    # near-duplicates (same data, different id) are left alone -- we can't
    # be sure they're the same person without a stronger identifier, so we
    # don't want to risk deleting real, distinct records.
    id_column = diag_cfg.get("id_column", "id")
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    print(f"[data_cleaning] step 4/7 - dropping exact duplicate rows: {n_dropped} row(s) removed")
    print(f"[data_cleaning]   shape after drop: {df.shape}")

    # --- step 5 (problem 4): out-of-range values -> NaN --------------------
    # can't guess the "right" value for e.g. age=-3 or priors_count=500, so
    # these are treated the same as any other missing value and left for
    # the imputation step (median / most_frequent) to handle later.
    validity_rules = diag_cfg.get("validity_rules", {})
    print(f"[data_cleaning] step 5/7 - flagging out-of-range values: {list(validity_rules.keys())}")
    for col, bounds in validity_rules.items():
        if col not in df.columns:
            print(f"[data_cleaning]   {col}: not in df, skipping")
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        lo = bounds.get("min", float("-inf"))
        hi = bounds.get("max", float("inf"))
        out_of_range_mask = numeric.notna() & ((numeric < lo) | (numeric > hi))
        n = int(out_of_range_mask.sum())
        if n > 0:
            examples = sorted(numeric[out_of_range_mask].unique().tolist())[:10]
            print(f"[data_cleaning]   {col}: {n} out-of-range value(s) -> NaN (rule={bounds}, examples={examples})")
        df[col] = numeric.where(~out_of_range_mask, pd.NA)

    # --- step 6 (problem 1, part a): MNAR "was missing" indicator columns --
    # must run BEFORE imputation below -- once a value is filled in with the
    # median/most-frequent, there's no way to tell afterwards that it used
    # to be missing.
    prep_cfg = config["preprocessing"]
    mnar_sources = prep_cfg.get("mnar_indicator_sources", [])
    print(f"[data_cleaning] step 6/7 - creating MNAR indicator columns for: {mnar_sources}")
    for col in mnar_sources:
        if col not in df.columns:
            print(f"[data_cleaning]   {col}: not in df, skipping")
            continue
        indicator_col = f"{col}_missing"
        df[indicator_col] = df[col].isna().astype(int)
        print(f"[data_cleaning]   created '{indicator_col}': {int(df[indicator_col].sum())} row(s) flagged as missing")

    # --- step 7 (problem 1, part b): impute remaining missing values -------
    # NOTE -- known simplification: these statistics (median / most frequent)
    # are computed over the WHOLE dataset, before the train/test split that
    # happens inside preprocess(). A fully leakage-free version would fit
    # them on X_train only and apply the same values to X_test. Left as-is
    # for now and flagged here on purpose -- revisit once encoding/scaling
    # is rebuilt as a proper sklearn Pipeline/ColumnTransformer (see
    # config.yaml's preprocessing.encoder/scaler), which handles train-only
    # fitting automatically.
    numeric_cols = prep_cfg.get("numeric_features", [])
    categorical_cols = prep_cfg.get("categorical_features", [])
    numeric_strategy = prep_cfg["imputation"]["numeric_strategy"]
    categorical_strategy = prep_cfg["imputation"]["categorical_strategy"]

    print(f"[data_cleaning] step 7/7 - imputing missing values "
          f"(numeric={numeric_strategy}, categorical={categorical_strategy})")
    print("[data_cleaning]   NOTE: fit on the full dataset, not just a future train split -- "
          "known simplification, see comment in code")

    for col in numeric_cols:
        if col not in df.columns:
            print(f"[data_cleaning]   {col}: not in df, skipping")
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        fill_value = df[col].median()
        df[col] = df[col].fillna(fill_value)
        print(f"[data_cleaning]   {col}: {n_missing} value(s) imputed with median = {fill_value}")

    for col in categorical_cols:
        if col not in df.columns:
            print(f"[data_cleaning]   {col}: not in df, skipping")
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        fill_value = df[col].mode(dropna=True).iloc[0]
        df[col] = df[col].fillna(fill_value)
        print(f"[data_cleaning]   {col}: {n_missing} value(s) imputed with most frequent = {fill_value!r}")

    print("[data_cleaning] finished, shape:", df.shape)
    return df


def diagnose(df: pd.DataFrame, config: dict) -> str:
    """
    Data-quality diagnostic report -- READ ONLY, does not modify `df`.

    Runs every check the class has flagged as relevant for this dataset
    (see config.yaml -> diagnostics) and prints/returns a text report:
      1. missing values per column, as loaded
      2. placeholder tokens hiding as real values (e.g. "-", "?", "N/A")
      3. inconsistent spelling/casing within categorical columns
      4. out-of-range / invalid numeric values
      5. exact vs. near-duplicate records
      6. redundant columns (near-perfectly correlated with another column)
      7. a first look at *why* values are missing (missingness mechanism)

    The point is to see the scale of each problem before deciding how
    to fix it -- this function only reports, it never cleans.
    """
    diag_cfg = config["diagnostics"]
    lines = []

    print("Número de linhas:", df.shape[0])
    print("Número de colunas:", df.shape[1])
    print(df.head())

    def header(title):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    # 1. missing values as loaded --------------------------------------
    header("1. Missing values per column (as loaded)")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
    if na_counts.empty:
        lines.append("  none")
    else:
        for col, n in na_counts.items():
            lines.append(f"  {col:<20s} {n:>5d}  ({n / len(df):.1%})")

    # 2. placeholder tokens hiding inside otherwise-loaded columns ------
    header("2. Placeholder tokens found (not yet NaN)")
    placeholder_tokens = set(diag_cfg.get("placeholder_tokens", []))
    placeholder_tokens.discard("")  # empty string is indistinguishable from real NaN on read
    any_found = False
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "str":
            vals = df[col].dropna().astype(str)
            found = vals[vals.isin(placeholder_tokens)]
            if len(found) > 0:
                any_found = True
                counts = found.value_counts().to_dict()
                lines.append(f"  {col:<20s} {dict(counts)}  (total {len(found)})")
    if not any_found:
        lines.append("  none")

    # 3. inconsistent categorical spelling/casing ------------------------
    header("3. Inconsistent category spellings (config's canonical_categories)")
    canonical = diag_cfg.get("canonical_categories", {})
    for col, mapping in canonical.items():
        if col not in df.columns:
            continue
        raw_vals = df[col].dropna().astype(str)
        normalized = raw_vals.str.strip().str.lower()
        unmapped_norms = set(normalized.unique()) - set(mapping.keys())
        # raw spellings that don't already equal their canonical target
        variant_counts = {}
        for raw, norm in zip(raw_vals, normalized):
            target = mapping.get(norm)
            if target is not None and raw != target:
                variant_counts[raw] = variant_counts.get(raw, 0) + 1
        lines.append(f"  {col}:")
        if variant_counts:
            for raw, n in sorted(variant_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {raw!r:<25s} -> {mapping[raw.strip().lower()]!r}  (n={n})")
        else:
            lines.append("    no variants outside canonical spelling")
        if unmapped_norms:
            lines.append(f"    NOT in canonical_categories mapping at all: {sorted(unmapped_norms)}")

    # 4. out-of-range / invalid numeric values ---------------------------
    header("4. Invalid values (config's validity_rules)")
    invalid_report = flag_invalid_values(df, diag_cfg.get("validity_rules", {}))
    for col, r in invalid_report.items():
        lines.append(f"  {col} (rule: {r['rule']}):")
        lines.append(f"    non-numeric entries: {r['n_non_numeric']}  examples={r['non_numeric_examples']}")
        lines.append(f"    out-of-range values:  {r['n_out_of_range']}  examples={r['out_of_range_examples']}")

    # 5. duplicates --------------------------------------------------------
    header("5. Duplicate records")
    dup_report = find_duplicates(df, diag_cfg.get("id_column", "id"))
    lines.append(f"  exact duplicates (same id, all columns identical): {dup_report['n_exact_duplicates']}"
                 f"  ({dup_report['n_exact_duplicate_rows_involved']} rows involved)")
    lines.append(f"    example ids: {dup_report['example_ids_exact']}")
    lines.append(f"  near-duplicates (identical data, different id):    {dup_report['n_near_duplicates']}"
                 f"  ({dup_report['n_near_duplicate_rows_involved']} rows involved)")
    lines.append(f"    example ids: {dup_report['example_ids_near']}")

    # 6. redundant columns ---------------------------------------------
    header("6. Redundant columns (config's redundant_columns)")
    for col in diag_cfg.get("redundant_columns", []):
        if col not in df.columns:
            lines.append(f"  {col}: not present")
            continue
        candidates = [c for c in df.columns if c != col and c not in diag_cfg.get("redundant_columns", [])]
        best_match, best_score = None, 0.0
        for other in candidates:
            a = pd.to_numeric(df[col], errors="coerce")
            b = pd.to_numeric(df[other], errors="coerce")
            if a.notna().sum() < 10 or b.notna().sum() < 10:
                continue
            corr = a.corr(b)
            if corr is not None and abs(corr) > best_score:
                best_score, best_match = abs(corr), other
        if best_match:
            lines.append(f"  {col:<18s} <-> {best_match:<18s}  correlation = {best_score:.3f}")
        else:
            lines.append(f"  {col}: no strongly correlated numeric column found")

    # 7. missingness mechanism (a first look at why values are missing) --
    header("7. Missingness mechanism (does missing-ness look random?)")
    miss_report = test_missingness_mechanism(
        df, target_col=config["data"]["target"], candidate_predictors=diag_cfg.get("candidate_predictors", [])
    )
    for col, r in miss_report.items():
        lines.append(f"  {col}  ({r['n_missing']} missing, {r['pct_missing']}%)")
        if "target_rate_when_missing" in r:
            lines.append(
                f"    two_year_recid rate | missing={r['target_rate_when_missing']}"
                f"  vs. present={r['target_rate_when_present']}"
            )
        if r["missing_rate_spread_by_predictor"]:
            lines.append(f"    missing-rate spread by predictor: {r['missing_rate_spread_by_predictor']}")

    text = "\n".join(lines)
    print(text)
    return text
"""
Preprocessing (week 4) -- cleaning rules, the dev/test split, and the leak-safe recipe.

Every step here is one of three kinds, and the kind decides WHERE it may run:

  stateless rule      clean_dataset(), add_missingness_indicators(), split_features_target()
                      Learns nothing and removes no rows -> runs on training data AND on new data.
  training-only step  drop_duplicate_rows()
                      Learns nothing, but REMOVES rows -> only on the labelled training file, before the split.
  fitted step         build_preprocessor()   (imputation, encoding, scaling)
                      Learns from data -> lives inside the model Pipeline, so it is fitted on the
                      TRAINING rows of each split / CV fold only and applied unchanged to everything else.

`config.yaml` stores rules ("median", "target encoding", "robust scaling"), never fitted values.

`race` never enters the model: it travels alongside the data (`extras`, with COMPAS's own
`score_text`) only so src/evaluate.py can audit fairness afterwards.
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)

INDICATOR_SUFFIX = "_was_missing"


# --------------------------------------------------------------------------------------
# Cleaning -- stateless rules, row-preserving
# --------------------------------------------------------------------------------------
def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Trims whitespace, maps every spelling variant to one canonical label, and turns placeholder tokens into NaN."""
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        # last step: listed tokens become NaN. "nan" must be one of them -- astype(str) above turns a
        # genuine NaN into the text "nan", and without this it would survive as a fake category.
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies `{column: {"min": ..., "max": ...}}` domain rules (either bound optional) and converts
    violations to NaN IN PLACE on `df`: "impossible but not missing" (an age of -3, a decile score
    of 23) is still missing. Returns a small report: violations found per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df[column] = numeric.mask(violations)
    return pd.DataFrame(report_rows)


def _flag_inconsistent_pairs(df: pd.DataFrame, checks: list) -> None:
    """
    Rows where a number and the category derived from it contradict each other (e.g. `age` vs
    `age_cat`). Each value is valid alone; the PAIR is not. We can't tell which one is wrong, so both
    become NaN (IN PLACE) and get imputed. Bounds are inclusive: a value on a bucket boundary is
    accepted by both neighbouring buckets.
    """
    for check in checks:
        numeric_col, category_col, ranges = check["numeric"], check["category"], check["ranges"]
        if numeric_col not in df.columns or category_col not in df.columns:
            continue
        values = pd.to_numeric(df[numeric_col], errors="coerce")
        consistent = pd.Series(True, index=df.index)
        for category, bounds in ranges.items():
            ok = pd.Series(True, index=df.index)
            if "min" in bounds:
                ok &= values >= bounds["min"]
            if "max" in bounds:
                ok &= values <= bounds["max"]
            consistent &= ~(df[category_col] == category) | ok
        bad = ~consistent & values.notna() & df[category_col].isin(list(ranges))
        df[numeric_col] = values.mask(bad)
        df.loc[bad, category_col] = np.nan


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the week 3 diagnosis: placeholder / domain-rule -> NaN conversion, category cleanup,
    contradictory pairs -> NaN, redundant-column removal. Target-agnostic, so it is safe on
    label-free data to predict.

    ROW-PRESERVING: every input row comes out, in the same order. Removing duplicate rows is a
    training-only decision and lives in `drop_duplicate_rows()` -- at prediction time every row needs
    a prediction.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    _flag_inconsistent_pairs(out, diagnostics_config.get("consistency_checks", []))

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY. Drops exact duplicate rows and repeated ids (keeping the first), so the same
    person can't be counted twice -- or land in both the development and the locked test set.
    Must run BEFORE `split_dev_test()`. Never call this on data you're predicting for.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


# --------------------------------------------------------------------------------------
# Features, target and the locked test set
# --------------------------------------------------------------------------------------
def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """
    Adds a `<col>_was_missing` flag for each MNAR-diagnosed column BEFORE that column is imputed, so a
    model can still see the pattern even though the fill value (median / mode) can't carry it.
    """
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}{INDICATOR_SUFFIX}"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is None and `extras` has no target when called on label-free data.
    `extras` = (sensitive attribute, COMPAS's score_text): for the fairness audit, never features.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    X = df[[c for c in df.columns if c not in always_drop]]
    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int, sensitive_attr: str = None):
    """
    Sets the FINAL TEST SET aside. Stratified split of X, y and extras together (row-aligned).
      - development set: everything we may learn from and compare models on; cross-validation
        (src/evaluate.py) splits it again into train / validation folds.
      - locked test set: never used to fit, tune, compare or choose anything. Size and seed live in
        config.yaml's `test_set` section and are not changed afterwards.

    Missing values of the sensitive attribute (audit only, never a feature) are filled with the
    DEVELOPMENT set's most frequent value, for both parts, so test rows never influence the fill.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    if sensitive_attr and sensitive_attr in extras_dev.columns:
        mode = extras_dev[sensitive_attr].mode()
        if len(mode):
            extras_dev = extras_dev.copy()
            extras_test = extras_test.copy()
            extras_dev[sensitive_attr] = extras_dev[sensitive_attr].fillna(mode.iloc[0])
            extras_test[sensitive_attr] = extras_test[sensitive_attr].fillna(mode.iloc[0])
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test


# --------------------------------------------------------------------------------------
# The recipe -- fitted steps, built unfitted, fitted inside the model Pipeline
# --------------------------------------------------------------------------------------
class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """'count' encoding: each category -> its share of the TRAINING rows (a category never seen -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.frequencies_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_) if not isinstance(X, pd.DataFrame) else X
        out = pd.DataFrame(index=X.index)
        for c in self.columns_:
            out[c] = X[c].map(self.frequencies_[c]).fillna(0.0).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def _sklearn_version() -> tuple:
    major, minor = sklearn.__version__.split(".")[:2]
    return int(major), int(minor)


def _target_encoder(seed):
    # scikit-learn's TargetEncoder is CROSS-FITTED: during fit each row is encoded with category means
    # computed WITHOUT that row's fold, so a row's own label never leaks into its own encoding.
    # From scikit-learn 1.9 the shuffling is set through a `cv` splitter (random_state is deprecated).
    if _sklearn_version() >= (1, 9):
        return TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", random_state=seed)


def _ordinal_encoder(prep_config: dict):
    # an explicit order per column when config gives it for ALL categorical columns; otherwise
    # alphabetical, which is arbitrary ("Greater than 45" would sit between the other two age buckets)
    order = prep_config.get("ordinal_categories", {})
    cols = prep_config["categorical_features"]
    categories = [order[c] for c in cols] if all(c in order for c in cols) else "auto"
    return OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

ENCODER_NAMES = ["onehot", "ordinal", "count", "target"]
SCALER_NAMES = list(_SCALERS)


def build_preprocessor(preprocessing_config: dict, encoder: str = None, scaler: str = None,
                       random_state: int = None) -> ColumnTransformer:
    """
    Factory: an UNFITTED ColumnTransformer for the encoder/scaler pair chosen in `config.yaml`'s
    `preprocessing` section (`encoder` / `scaler` / `random_state` arguments override it -- used by
    experiments/). Every encoder tolerates categories it has never seen. Nothing is fitted here:
    fitting happens later, on the training part of each split / CV fold only, because this object
    sits INSIDE the model's Pipeline (see main.py).

        numeric        : imputer -> scaler
        categorical    : imputer -> encoder
        _was_missing   : passed through (already 0/1)
    """
    encoder_name = encoder or preprocessing_config["encoder"]
    scaler_name = scaler or preprocessing_config["scaler"]
    seed = random_state if random_state is not None else preprocessing_config.get("random_state")
    if encoder_name not in ENCODER_NAMES:
        raise ValueError(f"Unknown encoder '{encoder_name}'. Options: {ENCODER_NAMES}")
    if scaler_name not in _SCALERS:
        raise ValueError(f"Unknown scaler '{scaler_name}'. Options: {SCALER_NAMES}")

    encoders = {
        "onehot": lambda: OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
        "ordinal": lambda: _ordinal_encoder(preprocessing_config),
        "count": lambda: FrequencyEncoder(),
        "target": lambda: _target_encoder(seed),
    }
    scaler_factory = _SCALERS[scaler_name]
    scaler_step = scaler_factory() if callable(scaler_factory) else scaler_factory
    imputation = preprocessing_config.get("imputation", {})

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler_step),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoders[encoder_name]()),
    ])
    indicator_cols = [f"{c}{INDICATOR_SUFFIX}" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, list(preprocessing_config["numeric_features"])),
         ("categorical", categorical_pipeline, list(preprocessing_config["categorical_features"])),
         ("indicators", "passthrough", indicator_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
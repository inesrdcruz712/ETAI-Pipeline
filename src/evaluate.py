"""
Evaluation -- stratified k-fold cross-validation of the WHOLE pipeline, and the reports built on it.

One train/validation split gives one number, and a different seed would give a different one.
Cross-validation splits the development set into k folds; every fold takes a turn as the validation
set while the pipeline (preprocessing + model) is fitted from scratch on the other k-1. That gives k
scores (mean +- std shows how much to trust them) and an OUT-OF-FOLD prediction for every
development row -- made by a model that never saw that row -- which the classification report and
the fairness check are computed from.

The locked test set is never touched here.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate

def build_cv(cv_config: dict) -> StratifiedKFold:
    """The stratified splitter described by config.yaml's `cv` section (same seed = same folds on every run)."""
    shuffle = cv_config.get("shuffle", True)
    return StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                           random_state=cv_config.get("random_state") if shuffle else None)

def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` on each fold's training part and scores it on that
    fold's validation part. Because `pipeline` contains the preprocessing too, imputation
    medians, encoder statistics and scaler means are re-learned inside every fold -- the
    validation fold never influences its own preprocessing.
    
    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold -- train score, validation score, and the gap between
        them (a large, consistent gap = overfitting).
      - y_oof: out-of-fold predictions. Every row gets a prediction from the one fold model
        that did NOT train on it, so all of them are "unseen" -- what the classification
        report and fairness check are computed on (more rows, and the locked test set stays
        untouched). Taken from the same fits as the scores, so nothing is fitted twice.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return fold_scores, y_oof


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy", title: str = None) -> str:
    """Per-fold table + mean +/- std, as text (printed, and saved to results/)."""
    lines = [
        title or f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    print(text)
    return text

def corrected_paired_test(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """
    Nadeau & Bengio's corrected paired t-test: is model A really better than model B, or is the
    difference within fold-to-fold noise?

    `scores_a` / `scores_b` are the two models' scores on the SAME folds (same cv.random_state), so
    they can be paired fold by fold; `n_train` / `n_test` are the rows in one fold's training / validation
    part. The plain paired t-test treats the k differences as independent, but CV folds share most of their
    training rows, so the differences are correlated and the plain test is far too confident. The correction
    inflates the variance by (1/k + n_test/n_train):

        t = mean(d) / sqrt((1/k + n_test/n_train) * var(d)),   df = k - 1,   two-sided p-value

    Returns a dict: mean_diff (A - B), std_diff, t, p_value, and `significant` (p < 0.05).
    Only call A "better" when `significant` is True; otherwise the lead is as large as the noise.
    """
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(d)
    if k < 2:
        raise ValueError("Need scores from at least 2 folds to compare two models.")
    var = d.var(ddof=1)
    mean = d.mean()
    if var == 0:
        # identical differences in every fold: no noise to compare against (t is 0 or infinite)
        t_stat = 0.0 if mean == 0 else np.inf * np.sign(mean)
        p_value = 1.0 if mean == 0 else 0.0
    else:
        t_stat = mean / np.sqrt((1.0 / k + n_test / n_train) * var)
        p_value = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    return {"mean_diff": float(mean), "std_diff": float(np.sqrt(var)), "t": float(t_stat),
            "p_value": float(p_value), "significant": bool(p_value < 0.05)}

def oof_classification_report(y_true, y_pred) -> str:
    """Classification report on the out-of-fold predictions."""
    text = "Classification report (out-of-fold predictions, development set):\n" + \
        classification_report(y_true, y_pred, zero_division=0)
    print(text)
    return text


def fairness_report(y_true, y_pred, extras: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that
    "accuracy" and "fair" are not the same thing. Computed on the OUT-OF-FOLD predictions.

    For each group, prints (and returns as text) the false positive rate (share of people who did NOT
    reoffend but were predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), same rows
    `y_true`, `y_pred` and `extras` must be row-aligned.
    """
    df = extras.reset_index(drop=True).copy()
    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = np.asarray(y_pred)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race (out-of-fold predictions)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
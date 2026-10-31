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
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate

def build_cv(cv_config: dict) -> StratifiedKFold:
    """The stratified splitter described by config.yaml's `cv` section (same seed = same folds on every run)."""
    shuffle = cv_config.get("shuffle", True)
    return StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                           random_state=cv_config.get("random_state") if shuffle else None)

def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers

def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
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
    fitted = cross_validate(pipeline, X, y, cv=cv, return_estimator=True, return_indices=True, n_jobs=n_jobs)

    rows = []
    oof_proba = np.empty(len(X))
    for fold, (model, train_idx, val_idx) in enumerate(
            zip(fitted["estimator"], fitted["indices"]["train"], fitted["indices"]["test"]), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        row = {"fold": fold}
        row["train"] = scorers[selection_metric](model, X_train, y_train)
        row["validation"] = scorers[selection_metric](model, X_val, y_val)
        row["gap"] = row["train"] - row["validation"]
        for name, scorer in scorers.items():
            row[name] = scorer(model, X_val, y_val)
        rows.append(row)
        oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_scores = pd.DataFrame(rows)
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table + mean +/- std, as text (printed, and saved to results/)."""
    lines = [
            f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
            "",
            fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
            "",
            fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
            f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
        ]
    text = "\n".join(lines)
    print(text)
    return text

def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
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
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}

def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race", threshold: float = 0.5) -> str:
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
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (out-of-fold predictions)",
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
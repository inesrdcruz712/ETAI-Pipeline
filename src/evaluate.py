"""
Evaluation (week 4) -- stratified k-fold cross-validation of the WHOLE pipeline.

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
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


def cross_validate_pipeline(pipeline, X_dev, y_dev, cv_config: dict):
    """
    Returns (fold_table, oof_pred): per-fold train / validation accuracy and their gap, and the
    out-of-fold predictions (a numpy array aligned with the rows of X_dev).

    `pipeline` is cloned for every fold, so nothing fitted in one fold can leak into the next.
    """
    shuffle = cv_config.get("shuffle", True)
    skf = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )

    oof_pred = np.empty(len(y_dev), dtype=int)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

        train_pred = fold_pipeline.predict(X_dev.iloc[train_idx])
        val_pred = fold_pipeline.predict(X_dev.iloc[val_idx])
        oof_pred[val_idx] = val_pred

        train_acc = accuracy_score(y_dev.iloc[train_idx], train_pred)
        val_acc = accuracy_score(y_dev.iloc[val_idx], val_pred)
        rows.append({"fold": fold, "train_acc": train_acc, "val_acc": val_acc, "gap": train_acc - val_acc})

    return pd.DataFrame(rows), oof_pred


def cv_report(fold_table: pd.DataFrame, y_dev, oof_pred) -> str:
    """Per-fold table + mean +- std + classification report on the out-of-fold predictions. Printed and returned."""
    k = len(fold_table)
    table = fold_table.copy()
    table["fold"] = table["fold"].astype(str)
    mean, std = fold_table[["train_acc", "val_acc", "gap"]].mean(), fold_table[["train_acc", "val_acc", "gap"]].std()
    table.loc[len(table)] = ["mean", mean["train_acc"], mean["val_acc"], mean["gap"]]
    table.loc[len(table)] = ["std", std["train_acc"], std["val_acc"], std["gap"]]

    lines = [
        f"Stratified {k}-fold cross-validation on the development set ({len(y_dev)} rows)",
        "(preprocessing + model are refitted from scratch inside every fold)",
        "",
        table.to_string(index=False, formatters={c: "{:+.3f}".format if c == "gap" else "{:.3f}".format
                                                 for c in ["train_acc", "val_acc", "gap"]}),
        "",
        f"Validation accuracy: {mean['val_acc']:.3f} +- {std['val_acc']:.3f}",
        f"Train - validation gap: {mean['gap']:+.3f} +- {std['gap']:.3f}   "
        "(a large positive gap means the model memorised the training rows)",
        "",
        "Classification report (out-of-fold predictions, every development row predicted by a model that never saw it):",
        classification_report(y_dev, oof_pred),
    ]
    text = "\n".join(lines)
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
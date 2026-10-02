"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Orchestrates the whole pipeline:
    load config -> load data -> clean (row-preserving) -> drop duplicates (training file only)
    -> features/target -> set the locked test set aside -> stratified k-fold cross-validation of
    preprocessing + model on the development set -> report (out-of-fold) + fairness check
    -> refit the final model on the whole development set -> save results
"""
import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    build_preprocessor, clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target,
)
from src.model import build_model
from src.evaluate import cross_validate_pipeline, cv_report, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_pipeline(config: dict) -> Pipeline:
    """Preprocessing + model as ONE estimator: fit() learns every fitted step from the rows it is given and nothing else."""
    return Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])


def main():
    config = load_config()
    data_cfg, prep_cfg = config["data"], config["preprocessing"]

    # the model may only ever see the columns listed in config -- never COMPAS's score, ids, race or the target
    forbidden = set(data_cfg["drop_columns"]) | {data_cfg["sensitive_attr"], data_cfg["target"]}
    clashes = forbidden & set(prep_cfg["numeric_features"] + prep_cfg["categorical_features"])
    if clashes:
        raise ValueError(f"These columns are configured as model features but must not be: {sorted(clashes)}")

    raw = load_data(data_cfg["path"])

    # 1. cleaning rules (row-preserving), then duplicates -- a training-file-only decision, before the split
    df = clean_dataset(raw, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"]["id_column"])

    # 2. features / target, then the locked test set is set aside and not touched again
    X, y, extras = split_features_target(df, data_cfg, prep_cfg["mnar_indicator_sources"])
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
        sensitive_attr=data_cfg["sensitive_attr"],
    )

    header = (
        f"Rows: {len(raw)} raw -> {len(df)} after removing duplicates  |  "
        f"development {len(X_dev)}  |  locked test set {len(X_test)}\n"
    )
    print(header)

    # 3. judge the whole pipeline by stratified k-fold cross-validation on the development set
    pipeline = make_pipeline(config)
    fold_table, oof_pred = cross_validate_pipeline(pipeline, X_dev, y_dev, config["cv"])
    report = header + "\n" + cv_report(fold_table, y_dev, oof_pred)
    report += "\n" + fairness_report(y_dev, oof_pred, extras_dev, sensitive_attr=data_cfg["sensitive_attr"])

    # 4. the final model: the same pipeline refit on ALL development rows (CV estimated how good it is; this is the model itself)
    final_model = make_pipeline(config).fit(X_dev, y_dev)
    report += (
        f"Final model ({config['model']['type']}) refit on all {len(X_dev)} development rows.\n"
        f"Locked test set: {len(X_test)} rows set aside -- NOT evaluated.\n"
    )
    print(report.splitlines()[-2])
    print(report.splitlines()[-1])

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")
    return final_model


if __name__ == "__main__":
    main()
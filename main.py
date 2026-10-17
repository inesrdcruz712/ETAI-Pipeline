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
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    build_preprocessor, clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target,
)
from src.model import build_model, build_pipeline
from src.evaluate import build_cv, cross_validate_pipeline, cv_report, fairness_report, oof_classification_report
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
    pipeline = build_pipeline(prep_cfg, config["model"])
    cv = build_cv(config["cv"])
    scoring = config["cv"].get("scoring", "accuracy")
    n_jobs = config["cv"].get("n_jobs", 1)
    
    tuning_config = config.get("tuning", {})
    tuning_on = tuning_config.get("enabled", False)
    
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs)
    title = (f"Cross-validation with the default hyperparameters ({len(fold_scores)} stratified folds, metric: {scoring})"
            if tuning_on else None)
    sections = [header, cv_report(fold_scores, scoring, title)]
    print()
    
    # 4. week 5: hyperparameter tuning -- on the development set only, every candidate scored by
    #    CV of the whole pipeline (see src/tuning.py)
    if tuning_on:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(
                f"tuning.enabled is true but config.yaml has no tuning.search_spaces entry for model type "
                f"'{model_type}' (it has: {list(search_spaces)}). Add one, or set tuning.enabled: false."
            )
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)
    
        # 4a. the honest estimate of "tune, then refit" (its out-of-fold predictions replace the untuned ones)
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        sections.append(cv_report(
            nested_scores, scoring,
            title=f"Nested cross-validation of the tuned {model_type} (outer: {len(nested_scores)} folds, "
                  f"inner: {tuning_config['n_splits']} folds, {n_trials} trials per fold; metric: {scoring})"))
        print()
    
    # 4b. the hyperparameters we keep: the same procedure, once, on the whole development set
    pipeline, study = tune_pipeline(
        pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
    sections.append(tuning_report(study, nested_scores, scoring))
    print()
    
    # 5. reports on the out-of-fold predictions (nested ones if we tuned)
    sections.append(oof_classification_report(y_dev, y_oof))
    print()
    sections.append(fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=data_cfg["sensitive_attr"]))

    # 6. the final model: the (tuned) pipeline refit on ALL development rows -- CV estimated how good the
    #    procedure is; this is the model itself
    final_model = pipeline.fit(X_dev, y_dev)
    closing = (
        f"Final model ({config['model']['type']}{', tuned' if tuning_on else ''}) refit on all {len(X_dev)} development rows.\n"
        f"Locked test set: {len(X_test)} rows set aside -- NOT evaluated.\n"
    )
    print(closing)
    sections.append(closing)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, "\n\n".join(sections))
    print(f"Full results saved to {path}")
    return final_model


if __name__ == "__main__":
    main()
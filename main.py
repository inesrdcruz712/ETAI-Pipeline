"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Orchestrates the whole pipeline:
    load config -> load data -> clean (row-preserving) -> drop duplicates (training file only)
    -> features/target -> set the locked test set aside
    -> stratified k-fold CV of preprocessing + model on the development set, on several metrics
       (the no-model floor is scored on the same folds)
    -> (if tuning.enabled) nested CV of the tuning procedure + tuning on the whole development set
    -> reports on the out-of-fold probabilities: all metrics, confusion matrix, fairness check
    -> refit the final (tuned) pipeline on the whole development set
    -> add the run to the scoreboard (results/runs.csv) and save the full report
"""
import os
import warnings
from datetime import datetime

# scikit-learn prints this notice once per parallel worker when the config (e.g. set_output) is not forwarded
# to joblib workers; nothing in this pipeline depends on it, and with n_jobs=-1 it floods the terminal. It has to
# be silenced in the worker processes too, hence the environment variable (inherited by joblib's workers).
os.environ["PYTHONWARNINGS"] = ",".join(filter(None, [os.environ.get("PYTHONWARNINGS"), "ignore::UserWarning:sklearn.utils.parallel"]))
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target
from src.model import build_pipeline
from src.evaluate import (
    build_cv, build_scorers, cross_validate_pipeline, cv_report, fairness_report, oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_cfg, prep_cfg = config["data"], config["preprocessing"]
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    # the model may only ever see the columns listed in config -- never COMPAS's score, ids, race or the target
    forbidden = set(data_cfg["drop_columns"]) | {data_cfg["sensitive_attr"], data_cfg["target"]}
    clashes = forbidden & set(prep_cfg["numeric_features"] + prep_cfg["categorical_features"])
    if clashes:
        raise ValueError(f"These columns are configured as model features but must not be: {sorted(clashes)}")

    df_raw = load_data(data_cfg["path"])

    # 1. cleaning rules (row-preserving), then duplicates -- a training-file-only decision, before the split
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    # 2. features / target, then the locked test set is set aside; from here on, every decision uses the
    #    development set only
    X, y, extras = split_features_target(df_clean, data_cfg, prep_cfg.get("mnar_indicator_sources", []))
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # 3. one metric drives decisions (tuning, ranking); every metric in `metrics` is reported
    cv_config = config["cv"]
    cv = build_cv(cv_config)
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    if selection_metric not in metrics:
        raise ValueError(f"cv.selection_metric '{selection_metric}' must be one of cv.metrics: {metrics}")
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away (not used)")

    # 4. the estimator (preprocessing + model, one Pipeline) on the default hyperparameters, and the
    #    no-model floor (a dummy) on the SAME folds
    pipeline = build_pipeline(prep_cfg, config["model"])
    floor_pipeline = build_pipeline(prep_cfg, {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    # 5. hyperparameter tuning: development set only, every candidate scored by CV of the whole pipeline
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}' (it has: {list(search_spaces)}). Add one, or set tuning.enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 5a. honest estimate on the same outer folds; its out-of-fold probabilities feed the reports
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # 5b. the hyperparameters that are kept: the same procedure, once, on all development rows
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<20s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    # 6. reports on the out-of-fold probabilities (the nested ones if we tuned)
    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=data_cfg["sensitive_attr"])

    # 7. the final model: the (tuned) pipeline refit on ALL development rows -- CV estimated how good the
    #    procedure is; this is the model itself
    pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    report += "\n" + refit + "\n"

    # 8. the scoreboard: one row per run, with the floor and the untuned score on the same folds
    row = {"run": run, "data": data_cfg["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": prep_cfg["encoder"], "scaler": prep_cfg["scaler"],
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")
    return pipeline


if __name__ == "__main__":
    main()
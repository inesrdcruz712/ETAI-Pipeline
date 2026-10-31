"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll
past it or close the window. This module writes the full report (cross-validation table, classification
report, fairness table) to a timestamped file in `results/` instead, so you can open it again later, or
compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime
import pandas as pd


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside `results_dir` (created automatically
    if it doesn't exist yet) and returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    prep, test_set, cv = config["preprocessing"], config["test_set"], config["cv"]
    tuning = config.get("tuning", {})
    if tuning.get("enabled", False):
        tuning_line = (f"Tuning: Optuna (TPE)  trials={tuning['n_trials']}  inner_folds={tuning['n_splits']}  "
                        f"random_state={tuning['random_state']}  (nested CV on the development set)\n")
    else:
        tuning_line = "Tuning: off\n"
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={prep['encoder']}  scaler={prep['scaler']}\n"
        f"Locked test set: size={test_set['size']}  random_state={test_set['random_state']}\n"
        f"Cross-validation: {cv['n_splits']} folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {cv['metrics']}  selection metric: {cv['selection_metric']}\n"
        + tuning_line
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path

def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs
"""Model construction."""
import inspect
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection

from src.preprocessing import build_preprocessor

def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    A forest of oblique trees: each tree is fit on a bootstrap sample, on `n_components`
    random sparse linear combinations of the features (a new set per tree), so its splits are
    hyperplanes (w1*x1 + w2*x2 + ... <= c) instead of one-feature steps. `tree_params` go to every
    tree (`max_features` defaults to "sqrt"). Tuned as `model__estimator__tree__<param>` and
    `model__estimator__project__<param>`.

    Built from two scikit-learn parts: SparseRandomProjection (the random combinations) feeding a
    DecisionTreeClassifier, bagged by BaggingClassifier (bootstrap + one random_state per tree, so
    every tree draws its own combinations).
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)

def adaboost(max_depth: int = 1, **params):
    """
    AdaBoost over decision trees of depth `max_depth` (1 = stumps); `params` go to
    AdaBoostClassifier. A function rather than the class itself, because the base tree is an object
    and config.yaml can only hold plain values. The trees' depth is tuned as `model__estimator__max_depth`.
    """
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)

_MODELS = {
    # always predicts the most frequent class -- the floor every real model must beat
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    # an ensemble of decision trees, each trained on a bootstrap sample of the rows with a random
    # subset of features at every split; predictions are averaged. Untuned for now.
    "random_forest": RandomForestClassifier,
    # the same, but every split is a random linear combination of the features (week 6)
        "oblique_random_forest": oblique_random_forest,
        # trees built one after another, each focused on the rows the previous ones got wrong (week 6)
        "adaboost": adaboost,
}

def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    # `type` and `params` in config.yaml go together: catch a params block written for another model
    # (e.g. a random forest's n_jobs left under decision_tree) with a readable message
    signature = inspect.signature(_MODELS[model_type])
    takes_any = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    unknown = [name for name in params if name not in signature.parameters]
    if unknown and not takes_any:
        raise ValueError(
            f"config.yaml -> model: type '{model_type}' does not accept the parameter(s) {unknown}. "
            f"Valid ones: {sorted(signature.parameters)}. Change `type` and `params` together "
            f"(examples in the comments under `model:`)."
        )

    return _MODELS[model_type](**params)

def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing and model as one
    Pipeline, so every learned step is re-fit on the training part of each fold. Step names
    address hyperparameters from outside (`model__max_depth`, `prep__numeric__impute__strategy`).
    Any new learned step (e.g. feature selection) belongs here, between "prep" and "model".
    """
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])

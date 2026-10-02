"""Model construction."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

_MODELS = {
    # always predicts the most frequent class -- the floor every real model must beat
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    # an ensemble of decision trees, each trained on a bootstrap sample of the rows with a random
    # subset of features at every split; predictions are averaged. Untuned for now.
    "random_forest": RandomForestClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)

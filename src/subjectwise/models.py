"""Classifier factory.

Every model is a ``StandardScaler`` + estimator :class:`~sklearn.pipeline.Pipeline`, so the scaler
is fitted on the training fold only and nothing about the held-out subject leaks into the model.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("lda", "svm_rbf", "svm_linear", "rf", "logreg")

MODEL_LABELS = {
    "lda": "LDA",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (linear)",
    "rf": "Random forest",
    "logreg": "Logistic reg.",
}


def make_model(name: str, random_state: int = 0) -> Pipeline:
    """Return an unfitted ``scaler -> classifier`` pipeline for one of :data:`MODEL_NAMES`."""
    if name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "svm_rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    elif name == "svm_linear":
        clf = SVC(kernel="linear", C=1.0, random_state=random_state)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    elif name == "logreg":
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])

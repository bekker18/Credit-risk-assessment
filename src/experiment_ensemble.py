"""Experiment: soft-voting ensemble of HGB + Logistic Regression.

The two models use DIFFERENT encodings of the PAY columns on purpose:
  - HGB  : PAY as ordinal numeric (build_preprocessor from preprocess.py)
  - LogReg: PAY one-hot encoded, with rare high delays (>=5) capped to 4 so the
            one-hot does not create ultra-rare columns that go missing in a CV fold.
Soft voting averages their predicted probabilities. Because the two models make
partly different errors, the average can beat either one alone.

Comparison experiment - does not overwrite the saved HGB model.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from preprocess import (
    CATEGORICAL,
    PAY_COLS,
    PAY_SUMMARY,
    build_preprocessor,
    load_data,
    split_X_y,
)

NUMERIC_LR = (
    ["LIMIT_BAL", "AGE"]
    + PAY_SUMMARY
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def cap_pay(X):
    # clip rare high delay levels down to 4 (named func so it is picklable for n_jobs)
    return np.clip(X, -2, 4)


def build_lr_preprocessor():
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    pay_pipe = Pipeline(
        [
            ("cap", FunctionTransformer(cap_pay)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_LR),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL,
            ),
            ("pay", pay_pipe, PAY_COLS),
        ],
        remainder="drop",
    )


df = load_data("data/raw/dev.csv")
X, y = split_X_y(df)
assert y is not None, "dev data must contain the target column"

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# the two members (HGB uses its tuned params from train.py)
hgb_pipe = Pipeline(
    [
        ("prep", build_preprocessor()),
        (
            "clf",
            HistGradientBoostingClassifier(
                random_state=42,
                class_weight="balanced",
                learning_rate=0.01,
                max_iter=500,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.1,
            ),
        ),
    ]
)
lr_pipe = Pipeline(
    [
        ("prep", build_lr_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)),
    ]
)

ensemble = VotingClassifier(
    estimators=[("hgb", hgb_pipe), ("lr", lr_pipe)], voting="soft"
)

# out-of-fold probabilities for the ensemble
proba = cross_val_predict(ensemble, X, y, cv=cv, method="predict_proba")[:, 1]

# default-threshold and tuned-threshold Macro F1
default_f1 = float(f1_score(y, (proba >= 0.5).astype(int), average="macro"))
best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.9, 0.01):
    f1 = float(f1_score(y, (proba >= t).astype(int), average="macro"))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Ensemble (HGB + LogReg, soft voting)")
print("  Macro F1 (default threshold):", round(default_f1, 4))
print("  Best threshold:", round(best_t, 2), "-> Macro F1:", round(best_f1, 4))
print("  ROC-AUC:", round(roc_auc_score(y, proba), 4))

preds = (proba >= best_t).astype(int)
print("\n", classification_report(y, preds))
print(confusion_matrix(y, preds))

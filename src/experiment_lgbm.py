"""Experiment: LightGBM gradient boosting.

Same family as HGB but with leaf-wise tree growth and richer regularisation, so it
is the model most likely to beat HGB outright on this kind of tabular data. Tree
based, so PAY stays ordinal numeric (reuses build_preprocessor).

Requires: pip install lightgbm  (uv add lightgbm)

Same protocol as the other experiments: stratified 5-fold CV, random search,
out-of-fold threshold tuning, Macro F1 + ROC-AUC. Does not save a model.
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_data, split_X_y

df = load_data("data/raw/dev.csv")
X, y = split_X_y(df)
assert y is not None, "dev data must contain the target column"

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline(
    [
        ("prep", build_preprocessor()),
        (
            "clf",
            LGBMClassifier(
                random_state=42,
                class_weight="balanced",
                subsample_freq=1,  # needed for the subsample (bagging) param to apply
                verbose=-1,  # silence LightGBM's training chatter
            ),
        ),
    ]
)

param_dist = {
    "clf__n_estimators": [200, 400, 600],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__num_leaves": [15, 31, 63],
    "clf__min_child_samples": [20, 50, 100],
    "clf__reg_lambda": [0.0, 0.1, 1.0],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=25,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
)
search.fit(X, y)

print("Best params:")
for k, v in search.best_params_.items():
    print(" ", k, "=", v)
print("Best CV Macro F1 (lightgbm):", round(search.best_score_, 4))

best = search.best_estimator_

# out-of-fold probabilities for threshold tuning
proba = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.9, 0.01):
    f1 = float(f1_score(y, (proba >= t).astype(int), average="macro"))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("\nBest threshold:", round(best_t, 2), "-> Macro F1:", round(best_f1, 4))
print("ROC-AUC:", round(roc_auc_score(y, proba), 4))

preds = (proba >= best_t).astype(int)
print("\n", classification_report(y, preds))
print(confusion_matrix(y, preds))

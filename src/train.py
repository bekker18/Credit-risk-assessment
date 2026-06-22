import warnings
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
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
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_data, split_X_y

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# load the development data
df = load_data("data/raw/dev.csv")
X, y = split_X_y(df)
assert y is not None, "dev data must contain the target column"

# stratified CV keeps the class ratio in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# baseline comparison: logistic regression vs LightGBM (default settings)
models = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "lgbm": LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1),
}
print("Model comparison (Macro F1):")
for name, m in models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", m)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    print(" ", name, round(scores.mean(), 4), "+/-", round(scores.std(), 4))

# LightGBM is our chosen model - tune its hyperparameters
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

# random search over the grid, scored on Macro F1
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

print("\nBest params:")
for k, v in search.best_params_.items():
    print(" ", k, "=", v)
print("Best CV Macro F1:", round(search.best_score_, 4))

best_pipe = search.best_estimator_

proba = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

# search for the threshold that gives the best Macro F1
best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.9, 0.01):
    f1 = float(f1_score(y, (proba >= t).astype(int), average="macro"))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("\nBest threshold:", round(best_t, 2), "-> Macro F1:", round(best_f1, 4))
print("ROC-AUC:", round(roc_auc_score(y, proba), 4))

# report at the tuned threshold
preds = (proba >= best_t).astype(int)
print("\n", classification_report(y, preds))
print(confusion_matrix(y, preds))

# save the best model + threshold
Path("models").mkdir(exist_ok=True)
joblib.dump({"model": best_pipe, "threshold": best_t}, "models/model.joblib")
print("\nModel saved to models/model.joblib")

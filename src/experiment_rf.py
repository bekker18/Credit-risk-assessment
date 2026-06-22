import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ]
)

# small random search over the main Random Forest knobs
param_dist = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 5, 20],
    "clf__max_features": ["sqrt", 0.5],
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=12,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
)
search.fit(X, y)

print("Best params:")
for k, v in search.best_params_.items():
    print(" ", k, "=", v)
print("Best CV Macro F1 (random forest):", round(search.best_score_, 4))

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

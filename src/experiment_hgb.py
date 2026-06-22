import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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

# load the development data
df = load_data("data/raw/dev.csv")
X, y = split_X_y(df)
assert y is not None, "dev data must contain the target column"

# stratified CV keeps the class ratio in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# models to try - both handle the imbalance with class_weight
models = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "hgb": HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
}

# quick comparison at the default threshold
print("Model comparison (Macro F1):")
for name, m in models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", m)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    print(" ", name, round(scores.mean(), 4), "+/-", round(scores.std(), 4))

# we go with gradient boosting and tune its hyperparameters
pipe = Pipeline(
    [
        ("prep", build_preprocessor()),
        (
            "clf",
            HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        ),
    ]
)

# parameter grid - keys are prefixed "clf__" because the model is a pipeline step
param_dist = {
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__max_iter": [200, 300, 500],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__min_samples_leaf": [20, 50, 100],
    "clf__l2_regularization": [0.0, 0.1, 1.0],
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

best_pipe = search.best_estimator_  # already refit on all data

# out-of-fold probabilities from the best config, to tune the threshold
proba = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

# search for the threshold that gives the best Macro F1
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

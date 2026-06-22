import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import CATEGORICAL, PAY_COLS, PAY_SUMMARY, load_data, split_X_y

# For logistic regression we move the PAY columns into the one-hot group.
NUMERIC_LR = (
    ["LIMIT_BAL", "AGE"]
    + PAY_SUMMARY
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
CATEGORICAL_LR = CATEGORICAL + PAY_COLS  # SEX, EDUCATION, MARRIAGE + PAY (one-hot)


def build_lr_preprocessor():
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_LR),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_LR,
            ),
        ],
        remainder="drop",
    )


df = load_data("data/raw/dev.csv")
X, y = split_X_y(df)
assert y is not None, "dev data must contain the target column"

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline(
    [
        ("prep", build_lr_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]
)

# tune the regularisation strength
grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
search.fit(X, y)

print("Best C:", search.best_params_["clf__C"])
print("Best CV Macro F1 (logreg, one-hot PAY):", round(search.best_score_, 4))

best = search.best_estimator_

# out-of-fold probabilities for threshold tuning
proba = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.9, 0.01):
    f1 = float(f1_score(y, (proba >= t).astype(int), average="macro"))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Best threshold:", round(best_t, 2), "-> Macro F1:", round(best_f1, 4))
print("ROC-AUC:", round(roc_auc_score(y, proba), 4))

preds = (proba >= best_t).astype(int)
print("\n", classification_report(y, preds))
print(confusion_matrix(y, preds))

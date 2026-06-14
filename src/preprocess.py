import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default.payment.next.month"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC = (
    ["LIMIT_BAL", "AGE"]
    + PAY_COLS
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def load_data(path):
    df = pd.read_csv(path)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def split_X_y(df):
    y = df[TARGET] if TARGET in df.columns else None
    X = df.drop(columns=[c for c in ["ID", TARGET] if c in df.columns])
    return X, y


def build_preprocessor():
    assert set(NUMERIC) | set(CATEGORICAL), "feature lists empty"

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default.payment.next.month"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# engineered repayment-status summaries (built from PAY_COLS in load_data)
PAY_SUMMARY = ["num_late", "max_delay", "mean_delay", "delay_trend"]

NUMERIC = (
    ["LIMIT_BAL", "AGE"]
    + PAY_COLS
    + PAY_SUMMARY
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def add_repayment_summaries(df):
    # condense the six PAY columns into a few strong distress signals.
    # row-wise only (no data-derived stats) so it is leakage-safe on dev and eval.
    pay = df[PAY_COLS]
    df["num_late"] = (pay >= 1).sum(axis=1)  # how many of the 6 months were late
    df["max_delay"] = pay.max(axis=1)  # worst delay across the 6 months
    df["mean_delay"] = pay.mean(axis=1)  # average delay level
    df["delay_trend"] = (
        df["PAY_0"] - df["PAY_6"]
    )  # recent minus oldest (>0 = worsening)
    return df


def load_data(path):
    df = pd.read_csv(path)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df = add_repayment_summaries(df)
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

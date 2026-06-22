import joblib
import pandas as pd

from preprocess import load_data, split_X_y

# load the trained model and the tuned threshold
bundle = joblib.load("models/model.joblib")
model = bundle["model"]
threshold = bundle["threshold"]

# load the evaluation data (no label column)
df = load_data("data/raw/eval.csv")
id_col = "ID" if "ID" in df.columns else "id"
ids = df[id_col]
X, _ = split_X_y(df)

# predict probabilities and apply the threshold
proba = model.predict_proba(X)[:, 1]
preds = (proba >= threshold).astype(int)

# write the submission file
submission = pd.DataFrame({"Id": ids, "Predicted": preds})
submission.to_csv("submission.csv", index=False)
print("Saved", len(submission), "predictions to submission.csv")

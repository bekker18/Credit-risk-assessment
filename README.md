
# Credit Card Default Prediction

## Project Overview

This project develops a machine learning pipeline for predicting whether a credit card client will default on their payment in the following month.

The task is formulated as a binary classification problem:

* `0` = No default
* `1` = Default

The main evaluation metric used for model selection is  **Macro F1-score** , as required by the project specification. Macro F1-score is appropriate for this task because the dataset is imbalanced and both classes should contribute equally to the final evaluation.

Only the dataset provided for the project is used. No external datasets are included.

---

## Project Structure

```text
.
├── data
│   ├── raw
│   │   ├── dev.csv
│   │   ├── eval.csv
│   │   └── submission.csv
│   └── processed
│       ├── correlation_num_features.png
│       ├── credit_limit_default.png
│       ├── default_distribution.png
│       └── default_rate_by_PAY_0.png
│
├── models
│   └── model.joblib
│
├── notebooks
│   └── EDA.ipynb
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── experiment_logreg.py
│   ├── experiment_rf.py
│   ├── experiment_lgbm.py
│   ├── experiment_hgb.py
│   └── experiment_ensemble.py
│
├── requirements.txt
├── README.md
└── submission.csv
```

---

## Environment Setup

Install the required dependencies from the project root directory:

```bash
pip install -r requirements.txt
```

The project was developed using Python and the following main libraries:

* pandas
* numpy
* scikit-learn
* LightGBM
* matplotlib
* joblib

---

## Data

The dataset files must be placed in the following directory:

```text
data/raw/
```

Expected input files:

```text
data/raw/dev.csv
data/raw/eval.csv
```

The development file contains the target column and is used for training and validation. The evaluation file does not contain the target column and is used to generate the final submission file.

---

## Preprocessing and Feature Engineering

The preprocessing pipeline is implemented in:

```text
src/preprocess.py
```

The preprocessing step includes:

* Loading the development and evaluation datasets
* Cleaning categorical values in `EDUCATION` and `MARRIAGE`
* Separating features and target
* Applying numerical imputation and scaling
* Applying one-hot encoding to categorical variables

Additional repayment-history features are created from the six repayment status columns:

* `num_late`: number of months with delayed payment
* `max_delay`: maximum repayment delay observed
* `mean_delay`: average repayment delay
* `delay_trend`: difference between the most recent and oldest repayment status

These features summarize the repayment behaviour of each client and are used to improve the predictive performance of the model.

---

## Training the Final Model

The final model training script is:

```text
src/train.py
```

Run the following command from the project root directory:

```bash
python src/train.py
```

The training script performs the following steps:

1. Loads the development dataset from `data/raw/dev.csv`
2. Applies preprocessing and feature engineering
3. Compares baseline models using Macro F1-score
4. Trains a LightGBM classifier
5. Performs hyperparameter tuning using `RandomizedSearchCV`
6. Uses stratified 5-fold cross-validation
7. Optimizes the classification threshold for Macro F1-score
8. Prints classification metrics and confusion matrix
9. Saves the final trained model and threshold to:

```text
models/model.joblib
```

---

## Generating the Submission File

After training the model, predictions for the evaluation set can be generated with:

```bash
python src/predict.py
```

The prediction script:

1. Loads the trained model from `models/model.joblib`
2. Loads the evaluation dataset from `data/raw/eval.csv`
3. Predicts default probabilities
4. Applies the optimized classification threshold
5. Saves the final submission file as:

```text
submission.csv
```

The generated file follows the required format:

```text
Id,Predicted
0,0
1,1
2,0
...
```

---

## Additional Experiments

Several additional experiment scripts are included to support model comparison and model selection.

### Logistic Regression

```bash
python src/experiment_logreg.py
```

### Random Forest

```bash
python src/experiment_rf.py
```

### HistGradientBoosting

```bash
python src/experiment_hgb.py
```

### LightGBM

```bash
python src/experiment_lgbm.py
```

### Soft-Voting Ensemble

```bash
python src/experiment_ensemble.py
```

These experiments evaluate different modelling strategies using stratified cross-validation and Macro F1-score.

---

## Exploratory Data Analysis

Exploratory data analysis is provided in:

```text
notebooks/EDA.ipynb
```

The notebook includes:

* Class distribution analysis
* Credit limit analysis
* Repayment status analysis
* Correlation analysis
* Visualizations used to support the report

Generated figures are stored in:

```text
data/processed/
```

---

## Reproducibility

Random seeds are fixed where applicable to make the experiments reproducible.

To reproduce the final pipeline from scratch, run:

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

This will train the final model and generate the final `submission.csv` file.

---

## Author

Bekzod Kadirov
Politecnico di Torino
Data Science and Engineering

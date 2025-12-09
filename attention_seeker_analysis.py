import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "attention_scores.csv")

# =========================
# DATA LOADING
# =========================

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. "
                                f"Make sure attention_scores.csv is in {BASE_DIR}.")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Try to parse datetime if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    print("Data loaded.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    return df


# =========================
# ORIGINAL REGRESSION:
# RECONSTRUCT ATTENTION SCORE
# =========================

def run_attention_score_regression(df: pd.DataFrame) -> None:
    """
    Regression that predicts attention_score from engineered features.
    This intentionally recovers the exact formula and acts as a correctness check.
    """

    if "attention_score" not in df.columns:
        print("\n[WARN] 'attention_score' not found – skipping regression.\n")
        return

    # Candidate features used in the original regression
    candidate_features = [
        "hr_mean",
        "hr_std",
        "movement_mean",
        "delta_hr",          # may or may not exist depending on the CSV version
        "delta_hrv",
        "delta_movement",
        "outside_factors",
    ]
    features = [f for f in candidate_features if f in df.columns]
    if not features:
        print("\n[WARN] No matching regression features found – skipping.\n")
        return

    X = df[features]
    y = df["attention_score"]

    # Use all data (no split) – this is a formula recovery, not generalization
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print("\n=== Regression Results (Attention Score Formula Check) ===")
    print("Features:", features)
    print(f"R^2:  {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("Coefficients:")
    for name, coef in zip(features, reg.coef_):
        print(f"  {name:16s}: {coef: .4f}")
    print(f"Intercept: {reg.intercept_: .4f}")


# =========================
# ORIGINAL CLASSIFICATION:
# ATTENTION LAPSE DETECTION
# =========================

def run_lapse_classifier(df: pd.DataFrame, threshold: float = -0.05) -> None:
    """
    Classify attention lapses based on attention_score threshold.
    """

    if "attention_score" not in df.columns:
        print("\n[WARN] 'attention_score' not found – skipping lapse classifier.\n")
        return

    df = df.copy()
    df["lapse"] = (df["attention_score"] < threshold).astype(int)

    print(f"\nLapse threshold: {threshold}")
    print("Lapse distribution:")
    print(df["lapse"].value_counts().rename("count"))

    candidate_features = [
        "hr_mean",
        "hr_std",
        "movement_mean",
        "delta_hr",
        "delta_hrv",
        "delta_movement",
        "outside_factors",
    ]
    features = [f for f in candidate_features if f in df.columns]
    if not features:
        print("\n[WARN] No features found for lapse classification – skipping.\n")
        return

    X = df[features]
    y = df["lapse"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    print("\n=== Classification Results (Lapse Detection) ===")
    print("Classification features:", features)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")


# =========================
# NEW TASK #1:
# COGNITIVE LOAD CLASSIFICATION
# =========================

def run_cognitive_load_classifier(df: pd.DataFrame) -> None:
    """
    New ML task #1:
    Predict cognitive load level ('level') from physiological features and outside_factors.
    """

    if "level" not in df.columns:
        print("\n[WARN] Column 'level' not found – skipping cognitive load classifier.\n")
        return

    clf_df = df.dropna(subset=["level"]).copy()

    # Candidate features – no attention_score
    candidate_features = [
        "hr_mean",
        "hr_std",
        "movement_mean",
        "hr_rest",
        "hrv_rest",
        "movement_rest",
        "delta_hrv",
        "delta_movement",
        "outside_factors",
    ]
    features_clf = [f for f in candidate_features if f in clf_df.columns]
    if not features_clf:
        print("\n[WARN] No classification features found – skipping cognitive load classifier.\n")
        return

    X = clf_df[features_clf]
    y = clf_df["level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print("\n=== Cognitive Load Classification (New Task) ===")
    print("Features used:", features_clf)
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


# =========================
# NEW TASK #2:
# ATTENTION REGRESSION WITHOUT DELTA FEATURES
# =========================

def run_attention_no_delta_regression(df: pd.DataFrame) -> None:
    """
    New ML task #2:
    Predict the continuous Attention Score WITHOUT using engineered delta features.
    This is a more realistic regression task than perfectly reconstructing the formula.
    """

    if "attention_score" not in df.columns:
        print("\n[WARN] Column 'attention_score' not found – skipping attention regression.\n")
        return

    candidate_features = [
        "hr_mean",
        "hr_std",
        "movement_mean",
        "hr_rest",
        "hrv_rest",
        "movement_rest",
        "outside_factors",
    ]
    features_reg = [f for f in candidate_features if f in df.columns]
    if not features_reg:
        print("\n[WARN] No regression features found – skipping attention regression.\n")
        return

    reg_df = df.dropna(subset=["attention_score"]).copy()
    X = reg_df[features_reg]
    y = reg_df["attention_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== Attention Regression WITHOUT Delta Features (New Task) ===")
    print("Features used:", features_reg)
    print(f"R^2:  {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("Coefficients:")
    for name, coef in zip(features_reg, reg.coef_):
        print(f"  {name:16s}: {coef: .4f}")
    print(f"Intercept: {reg.intercept_: .4f}")


# =========================
# MAIN
# =========================

def main():
    df = load_data(DATA_PATH)

    # Original sanity-check regression: recover Attention Score formula
    run_attention_score_regression(df)

    # Original lapse classifier
    run_lapse_classifier(df, threshold=-0.05)

    # New tasks
    run_cognitive_load_classifier(df)
    run_attention_no_delta_regression(df)


if __name__ == "__main__":
    main()

"""
Attention Seeker – End-to-end data pipeline

This script:
- Loads the merged sensor dataset from CogLoad1 (train/raw/merged_sensors.csv)
- Computes movement and an HRV proxy from heart-rate
- Computes baseline (rest) levels
- Aggregates data into 30-second windows per user
- Computes an Attention Score and delta HRV / delta movement
- Simulates Outside Factors (sleep, screen time) and aggregates them
- Exports:
    data/processed/attention_scores.csv
    data/processed/train.csv
    data/processed/test.csv

Assumes project root layout:

ML Final project/
  attention_seeker_pipeline.py
  data/
    ML_project_dataset/
      CogLoad1/
        train/
          raw/
            merged_sensors.csv
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Column definitions based on your actual CSV header
# ----------------------------------------------------------------------

TIME_COL = "datetime"      # primary time column (you also have 'timestamp', but we use 'datetime')
HR_COL = "hr"              # heart rate
ACC_X_COL = "band_ax"      # accelerometer X from band
ACC_Y_COL = "band_ay"      # accelerometer Y from band
ACC_Z_COL = "band_az"      # accelerometer Z from band
LEVEL_COL = "level"        # cognitive load / condition label
USER_COL = "user_id"       # subject ID


# ----------------------------------------------------------------------
# Data loading and preprocessing
# ----------------------------------------------------------------------

def load_raw(csv_path: Path) -> pd.DataFrame:
    """
    Load the merged_sensors.csv file and perform basic sanity checks.
    Ensures the required columns exist and parses the time column.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find merged_sensors.csv at: {csv_path}")

    print(f"[INFO] Loading raw data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # If 'datetime' is not present, fall back to 'timestamp'
    global TIME_COL
    if TIME_COL not in df.columns:
        if "timestamp" in df.columns:
            print("[WARN] 'datetime' not found; using 'timestamp' as TIME_COL")
            TIME_COL = "timestamp"
        else:
            raise ValueError(
                f"Expected a time column 'datetime' or 'timestamp' in {csv_path}. "
                f"Found columns: {list(df.columns)}"
            )

    # Check critical columns
    for col in [TIME_COL, HR_COL, ACC_X_COL, ACC_Y_COL, ACC_Z_COL, USER_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in {csv_path}, but it is not present. "
                f"Available columns: {list(df.columns)}"
            )

    # Parse time and sort
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    print("[INFO] Loaded", len(df), "rows after time parsing and sorting.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute movement and an HRV-like proxy from HR.
    movement = sqrt(ax^2 + ay^2 + az^2)
    hrv_proxy = |hr_t - hr_(t-1)|
    """
    print("[INFO] Preprocessing: computing movement and HRV proxy...")

    df = df.copy()  # avoid SettingWithCopy issues

    # Movement magnitude
    df["movement"] = np.sqrt(
        df[ACC_X_COL] ** 2 + df[ACC_Y_COL] ** 2 + df[ACC_Z_COL] ** 2
    )

    # HRV proxy: absolute difference of HR between successive samples
    df["hrv_proxy"] = df[HR_COL].diff().abs()

    # Drop rows where these are missing
    df = df.dropna(subset=[HR_COL, "movement", "hrv_proxy"])
    print("[INFO] Remaining rows after dropping NA:", len(df))

    return df


def compute_baselines(df: pd.DataFrame):
    """
    Compute rest (baseline) levels using medians:
    hr_rest, hrv_rest, movement_rest
    """
    print("[INFO] Computing baseline (rest) values...")
    eps = 1e-6

    hr_rest = max(df[HR_COL].median(), eps)
    hrv_rest = max(df["hrv_proxy"].median(), eps)
    movement_rest = max(df["movement"].median(), eps)

    print(f"    HR rest        = {hr_rest:.4f}")
    print(f"    HRV rest       = {hrv_rest:.4f}")
    print(f"    Movement rest  = {movement_rest:.4f}")

    return hr_rest, hrv_rest, movement_rest


# ----------------------------------------------------------------------
# Windowing and feature engineering
# ----------------------------------------------------------------------

def window_data(df: pd.DataFrame, window_seconds: int = 30) -> pd.DataFrame:
    """
    Aggregate sensor data into fixed-length windows per user.

    For each user_id & window:
      - datetime: minimum time in window
      - hr_mean: mean heart rate
      - hr_std: standard deviation of heart rate
      - movement_mean: mean of movement
      - hrv_mean: mean of hrv_proxy
      - level: first label in the window (string-safe)
    """
    print("[INFO] Windowing data into", window_seconds, "second windows...")

    df = df.copy()  # avoid SettingWithCopyWarning
    t0 = df[TIME_COL].min()

    # window index: floor((t - t0) / window_seconds)
    df["window_id"] = ((df[TIME_COL] - t0).dt.total_seconds() // window_seconds).astype(int)

    group_cols = [USER_COL, "window_id"]

    # Named aggregation; note 'level' uses 'first' (safe for string labels)
    agg_dict = {
        "datetime": (TIME_COL, "min"),
        "hr_mean": (HR_COL, "mean"),
        "hr_std": (HR_COL, "std"),
        "movement_mean": ("movement", "mean"),
        "hrv_mean": ("hrv_proxy", "mean"),
    }

    if LEVEL_COL in df.columns:
        # use the first observed label in the window, to avoid median() on strings
        agg_dict["level"] = (LEVEL_COL, "first")

    grouped = (
        df.groupby(group_cols)
          .agg(**agg_dict)
          .reset_index()
    )

    print("[INFO] Created", len(grouped), "windowed samples.")
    return grouped


def compute_attention(
    df_w: pd.DataFrame,
    hr_rest: float,
    hrv_rest: float,
    movement_rest: float,
) -> pd.DataFrame:
    """
    Compute Attention Score for each window using normalized deviations from baseline.

    HR term:  (hr_mean - hr_rest) / hr_rest
    HRV term: (hrv_mean - hrv_rest) / hrv_rest
    M term:   (movement_rest - movement_mean) / movement_rest

    Attention Score:
        0.25 * HR_term + 0.50 * HRV_term + 0.25 * M_term
    """
    print("[INFO] Computing attention features and Attention Score...")

    df_w = df_w.copy()
    eps = 1e-6
    hr_rest = max(hr_rest, eps)
    hrv_rest = max(hrv_rest, eps)
    movement_rest = max(movement_rest, eps)

    # Normalized terms
    df_w["HR_term"] = (df_w["hr_mean"] - hr_rest) / hr_rest
    df_w["HRV_term"] = (df_w["hrv_mean"] - hrv_rest) / hrv_rest
    df_w["M_term"] = (movement_rest - df_w["movement_mean"]) / movement_rest

    # Store delta_hrv and delta_movement explicitly (for analysis and matching your CSV header)
    df_w["delta_hrv"] = df_w["HRV_term"]
    df_w["delta_movement"] = df_w["M_term"]

    # Attention Score
    df_w["attention_score"] = (
        0.25 * df_w["HR_term"]
        + 0.50 * df_w["HRV_term"]
        + 0.25 * df_w["M_term"]
    )

    # Add baseline columns for export (constant per dataset)
    df_w["hr_rest"] = hr_rest
    df_w["hrv_rest"] = hrv_rest
    df_w["movement_rest"] = movement_rest

    return df_w


# ----------------------------------------------------------------------
# Outside Factors simulation
# ----------------------------------------------------------------------

def simulate_outside_factors(df_w: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Simulate daily sleep and screen-time, then aggregate them by window.
    OutsideFactors = (Sleep - 7.5) - (ScreenTime - 3.5)

    NOTE:
    We aggregate these synthetic factors at the window level, so the final
    attention_scores.csv only stores OutsideFactors (not the raw Sleep/Screen).
    """
    print("[INFO] Simulating Outside Factors (sleep + screen time)...")

    df_w = df_w.copy()
    rng = np.random.default_rng(seed)
    n = len(df_w)

    # Simulate per-window behavior
    sleep = rng.normal(7.5, 0.7, n)        # ~7.5 hours with some variation
    screen = rng.normal(3.5, 1.0, n)       # ~3.5 hours with some variation
    screen = np.clip(screen, 0.5, 8.0)     # clamp to [0.5, 8] hrs

    outside_factors = (sleep - 7.5) - (screen - 3.5)

    df_w["outside_factors"] = outside_factors
    return df_w


# ----------------------------------------------------------------------
# Export utilities
# ----------------------------------------------------------------------

def export_datasets(df_w: pd.DataFrame, output_dir: Path):
    """
    Export:
      - attention_scores.csv
      - train.csv
      - test.csv

    Columns in attention_scores.csv (matching your earlier header):

      datetime,
      hr_mean,
      hr_std,
      movement_mean,
      level,
      user_id,
      hr_rest,
      hrv_rest,
      movement_rest,
      delta_hrv,
      delta_movement,
      attention_score,
      outside_factors
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Exporting datasets to: {output_dir}")

    # Ensure level exists; if not, create a dummy label
    if "level" not in df_w.columns:
        df_w["level"] = "unknown"

    # Final columns for the main CSV
    cols = [
        "datetime",
        "hr_mean",
        "hr_std",
        "movement_mean",
        "level",
        USER_COL,          # 'user_id'
        "hr_rest",
        "hrv_rest",
        "movement_rest",
        "delta_hrv",
        "delta_movement",
        "attention_score",
        "outside_factors",
    ]

    missing = [c for c in cols if c not in df_w.columns]
    if missing:
        raise ValueError(f"Missing expected columns in df_w before export: {missing}")

    final_df = df_w[cols].copy()

    # Save full attention_scores.csv
    attention_path = output_dir / "attention_scores.csv"
    final_df.to_csv(attention_path, index=False)
    print(f"[INFO] Wrote attention_scores.csv with {len(final_df)} rows")

    # Train/test split for ML
    try:
        train_df, test_df = train_test_split(
            final_df,
            test_size=0.25,
            random_state=42,
            stratify=final_df["level"],
        )
    except Exception as e:
        print("[WARN] Stratified split failed, falling back to non-stratified:", e)
        train_df, test_df = train_test_split(
            final_df,
            test_size=0.25,
            random_state=42,
        )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Wrote train.csv with {len(train_df)} rows")
    print(f"[INFO] Wrote test.csv with  {len(test_df)} rows")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def main():
    # Project root = directory containing this script
    base_dir = Path(__file__).resolve().parent

    # Raw merged sensors path (train side)
    raw_path = (
        base_dir
        / "data"
        / "ML_project_dataset"
        / "CogLoad1"
        / "train"
        / "raw"
        / "merged_sensors.csv"
    )

    # Output directory for processed CSVs
    output_dir = base_dir / "data" / "processed"

    print("[INFO] Base directory:   ", base_dir)
    print("[INFO] Raw data path:    ", raw_path)
    print("[INFO] Output directory: ", output_dir)

    # 1. Load
    df_raw = load_raw(raw_path)

    # 2. Preprocess (movement, hrv_proxy)
    df_pre = preprocess(df_raw)

    # 3. Baselines
    hr_rest, hrv_rest, movement_rest = compute_baselines(df_pre)

    # 4. Windowing
    df_w = window_data(df_pre, window_seconds=30)

    # 5. Attention Score
    df_w = compute_attention(df_w, hr_rest, hrv_rest, movement_rest)

    # 6. Outside Factors
    df_w = simulate_outside_factors(df_w, seed=42)

    # 7. Export
    export_datasets(df_w, output_dir)

    print("[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    main()

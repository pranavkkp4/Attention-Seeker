"""
attention_seeker_analysis.py
================================

This script implements a simplified version of the ``Attention Seeker`` research
project.  It processes the publicly released cognitive load dataset (CogLoad1)
and computes a derived ``Attention Score`` for each 30‑second window of
recorded data.  The Attention Score captures how a user’s heart rate,
heart‑rate variability and movement differ from their resting baselines.

The script also demonstrates how one might simulate an ``Outside Factors``
metric (e.g. sleep and screen time) and compute the Pearson correlation
between Outside Factors and the Attention Score.  Finally, it trains
example regression and classification models to show that basic machine
learning techniques can predict the Attention Score and cognitive load
labels from the physiological features.

The overall workflow is:

1.  Read the merged sensor CSV provided in the CogLoad1 dataset.
2.  Parse timestamps and compute a movement magnitude from the three
    accelerometer axes.
3.  Resample each subject’s data into 30‑second windows and compute
    aggregated features (mean heart rate, heart‑rate variability as the
    standard deviation of heart rate within the window, and mean movement
    magnitude).
4.  Determine a per‑subject baseline for heart rate, heart‑rate
    variability and movement by averaging over all available data.
5.  Compute the Attention Score as a weighted combination of the
    fractional change from baseline for each metric:

       AttentionScore = 0.25 * ΔHR + 0.50 * ΔHRV + 0.25 * ΔM,

    where ΔHR = (HR_mean – HR_rest) / HR_rest,
          ΔHRV = (HRV_std – HRV_rest) / HRV_rest,
          ΔM  = (M_rest – M_mean) / M_rest.

6.  Simulate an Outside Factors score for demonstration purposes.  Since
    sleep and screen‑time data are not included in the released dataset,
    this script simply generates a random normal variable centred on zero
    for each window.  In a real application, one would replace this with
    actual sleep duration and screen‑time logs.
7.  Compute the Pearson correlation coefficient between the Attention
    Score and the Outside Factors score and report its value.
8.  Train a linear regression model to predict the Attention Score from
    the physiological features and report the coefficient of determination
    (R²) and mean absolute error (MAE).
9.  Train a logistic regression classifier to predict the cognitive load
    label (0/1/2) from the same features and report accuracy, F1 and
    ROC‑AUC scores.
10. Save a scatter plot of Attention Score versus Outside Factors to
    ``attention_scatter.png`` and write the processed windowed dataset to
    ``attention_scores.csv`` in the project folder.

Requirements:
    - Python 3.7+
    - pandas
    - numpy
    - scikit‑learn
    - matplotlib

To run the script from the command line:

    python attention_seeker_analysis.py --data-path PATH_TO_MERGED_CSV --output-dir OUTPUT_DIRECTORY

If you omit ``--data-path`` it defaults to ``data/merged_sensors.csv``.  If
``--output-dir`` is omitted, results are stored in the current working
directory.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")  # Use non‑interactive backend
import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the merged sensors CSV into a DataFrame.

    Parameters
    ----------
    csv_path: Path
        Path to the merged sensors CSV file.

    Returns
    -------
    DataFrame
        Raw sensor readings with parsed timestamps.
    """
    df = pd.read_csv(csv_path)
    # Parse the datetime string into a proper datetime object
    df["datetime"] = pd.to_datetime(df["datetime"])
    # Drop rows where heart rate or accelerometer data is missing
    df = df.dropna(subset=["hr", "band_ax", "band_ay", "band_az"])
    # Cast user_id to string to preserve leading zeros if any
    df["user_id"] = df["user_id"].astype(str)
    return df


def compute_movement_magnitude(df: pd.DataFrame) -> pd.Series:
    """Compute the Euclidean norm of the accelerometer axes.

    Parameters
    ----------
    df: DataFrame
        DataFrame with columns `band_ax`, `band_ay` and `band_az`.

    Returns
    -------
    Series
        Magnitude of movement for each row.
    """
    return np.sqrt(df["band_ax"]**2 + df["band_ay"]**2 + df["band_az"]**2)


def window_and_aggregate(df: pd.DataFrame, window: str = "30S") -> pd.DataFrame:
    """Aggregate sensor data into fixed windows for each user.

    The function resamples each user’s time series into non‑overlapping
    windows of a specified duration and computes mean heart rate, heart‑rate
    variability (standard deviation of heart rate), and mean movement.

    Parameters
    ----------
    df: DataFrame
        Raw sensor data with `datetime`, `user_id`, `hr` and movement
        magnitude columns.
    window: str, optional
        Window size in pandas offset alias format (default: "30S" for
        30 seconds).

    Returns
    -------
    DataFrame
        Aggregated features per user per window.
    """
    # Compute movement magnitude and add as a column
    df = df.copy()
    df["movement"] = compute_movement_magnitude(df)
    # Set datetime as index for resampling
    df = df.set_index("datetime")
    # Group by user and resample
    agg_dfs = []
    for user_id, user_df in df.groupby("user_id"):
        # Resample into fixed windows
        resampled = user_df.resample(window).agg(
            hr_mean=("hr", "mean"),
            hr_std=("hr", "std"),
            movement_mean=("movement", "mean"),
            level=("level", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        )
        resampled["user_id"] = user_id
        agg_dfs.append(resampled.dropna(subset=["hr_mean", "hr_std", "movement_mean"]))
    aggregated = pd.concat(agg_dfs)
    aggregated.reset_index(inplace=True)
    return aggregated


def compute_baselines(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Compute per‑user resting baselines.

    For each user, compute the mean of the aggregated statistics across all
    windows.  These serve as resting values for heart rate, heart‑rate
    variability and movement.

    Parameters
    ----------
    aggregated: DataFrame
        Aggregated window features with columns `user_id`, `hr_mean`,
        `hr_std` and `movement_mean`.

    Returns
    -------
    DataFrame
        Baseline statistics indexed by user_id.
    """
    baseline = aggregated.groupby("user_id").agg(
        hr_rest=("hr_mean", "mean"),
        hrv_rest=("hr_std", "mean"),
        movement_rest=("movement_mean", "mean"),
    )
    return baseline


def compute_attention_score(aggregated: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Add the Attention Score to the aggregated DataFrame.

    Parameters
    ----------
    aggregated: DataFrame
        Aggregated window features.
    baseline: DataFrame
        Per‑user baseline values for heart rate, heart‑rate variability and
        movement.

    Returns
    -------
    DataFrame
        Aggregated window features with additional columns for delta HR,
        delta HRV, delta movement and the Attention Score.
    """
    # Merge baseline into aggregated on user_id
    merged = aggregated.merge(baseline, left_on="user_id", right_index=True, how="left")
    # Compute fractional changes
    merged["delta_hr"] = (merged["hr_mean"] - merged["hr_rest"]) / merged["hr_rest"]
    merged["delta_hrv"] = (merged["hr_std"] - merged["hrv_rest"]) / merged["hrv_rest"]
    merged["delta_movement"] = (merged["movement_rest"] - merged["movement_mean"]) / merged["movement_rest"]
    # Weighted sum: 0.25*HR + 0.5*HRV + 0.25*M
    merged["attention_score"] = 0.25 * merged["delta_hr"] + 0.50 * merged["delta_hrv"] + 0.25 * merged["delta_movement"]
    return merged


def simulate_outside_factors(n: int, seed: int = 42) -> np.ndarray:
    """Generate a random outside factors score.

    Since sleep and screen‑time data are not part of the released
    dataset, this helper returns normally distributed values centred on zero
    as a placeholder.  For real use, replace this logic with actual
    measurements.

    Parameters
    ----------
    n: int
        Number of values to generate.
    seed: int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    ndarray
        Array of simulated outside factors scores.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=n)


def regression_model(X: pd.DataFrame, y: pd.Series):
    """Train a simple linear regression model and return evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae


def classification_model(X: pd.DataFrame, y: pd.Series):
    """Train a multinomial logistic regression classifier and return evaluation metrics."""
    # Filter rows where y is one of the numeric classes '0', '1' or '2'
    mask = y.isin(['0', '1', '2', 0, 1, 2])
    X = X[mask]
    y = y[mask].astype(int)
    if y.nunique() < 2:
        # Not enough classes to train
        return float('nan'), float('nan'), float('nan')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = LogisticRegression(multi_class='multinomial', max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        y_prob = clf.predict_proba(X_test)
        auc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr')
    except Exception:
        auc = float('nan')
    return acc, f1, auc


def plot_attention_vs_outside(att_scores: pd.Series, outside_factors: pd.Series, output_path: Path):
    """Create a scatter plot of Attention Score vs Outside Factors."""
    plt.figure(figsize=(6, 4))
    plt.scatter(outside_factors, att_scores, alpha=0.5)
    plt.title('Attention Score vs Outside Factors')
    plt.xlabel('Outside Factors (simulated)')
    plt.ylabel('Attention Score')
    # Fit a regression line for visualisation
    m, b = np.polyfit(outside_factors, att_scores, 1)
    x_vals = np.linspace(outside_factors.min(), outside_factors.max(), 100)
    plt.plot(x_vals, m * x_vals + b, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute Attention Scores and run ML models.')
    parser.add_argument('--data-path', type=str, default='data/merged_sensors.csv',
                        help='Path to merged_sensors.csv')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to write outputs')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data…')
    df = load_data(data_path)
    # Window and aggregate features
    print('Aggregating data into windows…')
    aggregated = window_and_aggregate(df)
    # Compute baselines per user
    print('Computing baselines…')
    baseline = compute_baselines(aggregated)
    # Compute attention score
    print('Computing attention scores…')
    scored = compute_attention_score(aggregated, baseline)
    # Simulate outside factors
    print('Simulating outside factors…')
    scored['outside_factors'] = simulate_outside_factors(len(scored))
    # Compute correlation
    corr = scored['attention_score'].corr(scored['outside_factors'])
    print(f'Pearson correlation between Attention Score and Outside Factors: {corr:.3f}')
    # Save scatter plot
    scatter_path = output_dir / 'attention_scatter.png'
    print(f'Saving scatter plot to {scatter_path}')
    plot_attention_vs_outside(scored['attention_score'], scored['outside_factors'], scatter_path)
    # Save processed data
    csv_out = output_dir / 'attention_scores.csv'
    print(f'Saving processed data to {csv_out}')
    scored.to_csv(csv_out, index=False)
    # Prepare feature matrix for models
    features = scored[['hr_mean', 'hr_std', 'movement_mean', 'delta_hr', 'delta_hrv', 'delta_movement', 'outside_factors']]
    # Regression: predict attention score
    print('Training regression model…')
    r2, mae = regression_model(features, scored['attention_score'])
    print(f'Regression R²: {r2:.3f}, MAE: {mae:.3f}')
    # Classification: predict cognitive load level
    print('Training classification model…')
    acc, f1, auc = classification_model(features, scored['level'])
    print(f'Classification accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')

    # Write a summary report
    summary_path = output_dir / 'report.txt'
    with open(summary_path, 'w') as f:
        f.write('Attention Seeker Project Report\n')
        f.write('===============================\n\n')
        f.write(f'Number of windows: {len(scored)}\n')
        f.write(f'Pearson correlation between Attention Score and Outside Factors: {corr:.3f}\n')
        f.write(f'Regression R²: {r2:.3f}, MAE: {mae:.3f}\n')
        f.write(f'Classification accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}\n')
    print(f'Summary written to {summary_path}')

if __name__ == '__main__':
    main()
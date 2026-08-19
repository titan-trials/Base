"""
Calibration checking -- a genuinely different question than AUC/accuracy.
AUC asks "does the model rank things correctly." Calibration asks "when
the model says 65%, does the outcome actually happen about 65% of the
time across many such predictions." A model can have great AUC and still
be badly calibrated (overconfident or underconfident), so this is worth
checking independently, not inferred from AUC.

Method: bucket predictions by their predicted probability (e.g. all
predictions between 60-70%), then for each bucket compare the AVERAGE
predicted probability against the ACTUAL observed frequency of the
outcome in that bucket. Good calibration = the two numbers stay close
across every bucket.
"""
import pandas as pd


def calibration_table(y_true, probs, n_bins: int = 10) -> pd.DataFrame:
    """
    y_true: actual 0/1 outcomes
    probs: predicted probabilities from the model
    n_bins: how many buckets to split predictions into (10 = deciles)

    Returns a DataFrame with one row per bucket: predicted range, mean
    predicted probability, actual observed rate, and sample count. Small
    counts in a bucket make that row's "actual rate" unreliable -- treat
    buckets with under ~20-30 samples with real skepticism.
    """
    df = pd.DataFrame({"y_true": y_true, "prob": probs})
    df["bucket"] = pd.cut(df["prob"], bins=n_bins, include_lowest=True)

    table = df.groupby("bucket", observed=True).agg(
        mean_predicted=("prob", "mean"),
        actual_rate=("y_true", "mean"),
        count=("y_true", "size"),
    ).reset_index()

    table["gap"] = table["actual_rate"] - table["mean_predicted"]
    return table


def print_calibration_report(y_true, probs, n_bins: int = 10, label: str = "Model"):
    table = calibration_table(y_true, probs, n_bins=n_bins)
    print(f"\n=== Calibration Report: {label} ===")
    print(f"{'Predicted Range':<20} {'Mean Pred':>10} {'Actual Rate':>12} {'Count':>8} {'Gap':>8}")
    for _, row in table.iterrows():
        low_confidence = " (small sample -- low confidence)" if row["count"] < 30 else ""
        print(
            f"{str(row['bucket']):<20} {row['mean_predicted']:>10.3f} "
            f"{row['actual_rate']:>12.3f} {row['count']:>8.0f} {row['gap']:>+8.3f}{low_confidence}"
        )
    print(
        "\nGap close to 0 = well-calibrated at that probability level. "
        "Consistently positive gap = model is UNDERconfident (real rate higher than predicted). "
        "Consistently negative gap = model is OVERconfident (real rate lower than predicted)."
    )
"""Create train/validation/holdout summaries for round 14 daily IC output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PERIODS = {
    "train_2017_2022": (pd.Timestamp("2017-01-01"), pd.Timestamp("2022-12-31")),
    "validation_2023_2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "holdout_2025_2026": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")),
    "full_2017_2026": (pd.Timestamp("2017-01-01"), pd.Timestamp("2026-12-31")),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-ic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    daily = pd.read_parquet(args.daily_ic)
    daily["TRADE_DATE"] = pd.to_datetime(daily["TRADE_DATE"]).dt.normalize()
    rows = []
    for factor, group in daily.groupby("factor", sort=True):
        for period, (start, end) in PERIODS.items():
            values = pd.to_numeric(
                group.loc[group["TRADE_DATE"].between(start, end), "neutral_ic"],
                errors="coerce",
            ).dropna()
            mean = float(values.mean()) if len(values) else np.nan
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append(
                {
                    "factor": factor,
                    "period": period,
                    "mean_ic": mean,
                    "abs_mean_ic": abs(mean) if np.isfinite(mean) else np.nan,
                    "icir": mean / std if np.isfinite(std) and std > 0 else np.nan,
                    "positive_rate": float(values.gt(0).mean()) if len(values) else np.nan,
                    "days": len(values),
                }
            )
    result = pd.DataFrame(rows)
    wide = result.pivot(index="factor", columns="period", values="mean_ic").reset_index()
    full_stats = result.loc[result["period"].eq("full_2017_2026"), [
        "factor", "icir", "positive_rate", "days"
    ]]
    wide = wide.merge(full_stats, on="factor", how="left")
    wide["stable_all_periods_positive"] = wide[
        ["train_2017_2022", "validation_2023_2024", "holdout_2025_2026"]
    ].gt(0).all(axis=1)
    wide = wide.sort_values("full_2017_2026", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()

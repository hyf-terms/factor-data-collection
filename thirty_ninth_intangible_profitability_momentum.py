"""Round 39: PIT-safe momentum of intangible-adjusted profitability.

The accounting characteristic is the round-38 TTM operating-profitability
measure with R&D and 30% of SG&A added back.  This script adds only temporal
information from that same characteristic: its one-year change and the change
in that change.  Cross-sectional robust z-scores put level and changes on a
common scale.  All weights and lags are fixed ex ante and no label is read.

Stocks without a full lag history receive the same-date neutral value.  No
stock or trading date is removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["TRADE_DATE", "SECURITY_ID"]
SOURCE = "r38_ttm_op_plus_rd_30sga_assets"
OUTPUT_COLUMNS = [
    "r39_intang_profit_level_z",
    "r39_intang_profit_yoy_change_z",
    "r39_intang_profit_acceleration_z",
    "r39_intang_profit_level_yoy_equal",
    "r39_intang_profit_level_yoy_70_30",
    "r39_intang_profit_level_yoy_accel_equal",
]


def _robust_z(data: pd.DataFrame, value: pd.Series) -> pd.Series:
    dates = data["TRADE_DATE"]
    value = pd.to_numeric(value, errors="coerce")
    median = value.groupby(dates, sort=False).transform("median")
    deviation = (value - median).abs()
    mad = deviation.groupby(dates, sort=False).transform("median") * 1.4826
    fallback = value.groupby(dates, sort=False).transform("std")
    scale = mad.where(mad.gt(1e-12), fallback.where(fallback.gt(1e-12)))
    return ((value - median) / scale).clip(-5.0, 5.0).fillna(0.0).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual-lag", type=int, default=252)
    args = parser.parse_args()

    data = pd.read_parquet(args.input, columns=KEYS + [SOURCE])
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"]).dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"]).astype("int64")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"]).drop_duplicates(KEYS, keep="last")

    level = pd.to_numeric(data[SOURCE], errors="coerce")
    grouped = level.groupby(data["SECURITY_ID"], sort=False)
    lag1 = grouped.shift(args.annual_lag)
    lag2 = grouped.shift(2 * args.annual_lag)
    yoy = level - lag1
    acceleration = level - 2.0 * lag1 + lag2

    data["r39_intang_profit_level_z"] = _robust_z(data, level)
    data["r39_intang_profit_yoy_change_z"] = _robust_z(data, yoy)
    data["r39_intang_profit_acceleration_z"] = _robust_z(data, acceleration)
    level_z = data["r39_intang_profit_level_z"]
    yoy_z = data["r39_intang_profit_yoy_change_z"]
    accel_z = data["r39_intang_profit_acceleration_z"]
    data["r39_intang_profit_level_yoy_equal"] = 0.50 * level_z + 0.50 * yoy_z
    data["r39_intang_profit_level_yoy_70_30"] = 0.70 * level_z + 0.30 * yoy_z
    data["r39_intang_profit_level_yoy_accel_equal"] = (level_z + yoy_z + accel_z) / 3.0
    data[OUTPUT_COLUMNS] = data[OUTPUT_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data[OUTPUT_COLUMNS] = data[OUTPUT_COLUMNS].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.sort_values(KEYS)[KEYS + OUTPUT_COLUMNS].to_parquet(
        args.output, index=False, compression="zstd"
    )
    print(f"saved {len(data):,} rows and {len(OUTPUT_COLUMNS)} factors to {args.output}")


if __name__ == "__main__":
    main()

"""Economically justified dense versions of cumulative guarantee balances."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dense_q1_gross_profit_factors import robust_daily_zscore
from event_financial_factor_search import KEYS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cols = ["r55_low_external_guarantee_assets", "r55_low_related_guarantee_assets"]
    d = pd.read_parquet(a.input, columns=KEYS + cols)
    # The source is a cumulative guarantee table.  No active cumulative record
    # means no reported balance, so zero is an economic state rather than a
    # cross-sectional median imputation.
    d[cols] = d[cols].fillna(0.0)
    z = robust_daily_zscore(d, cols)
    d["r55_dense_low_external_guarantee_assets"] = d[cols[0]]
    d["r55_dense_low_related_guarantee_assets"] = d[cols[1]]
    d["r55_dense_low_guarantee_equal"] = 0.5 * z[cols[0]] + 0.5 * z[cols[1]]
    d["r55_dense_low_guarantee_external70"] = 0.7 * z[cols[0]] + 0.3 * z[cols[1]]
    factors = [c for c in d if c.startswith("r55_dense_")]
    d[factors] = d[factors].astype("float32")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    d[KEYS + factors].to_parquet(a.output, index=False, compression="zstd")
    print(d[factors].isna().mean().to_string())


if __name__ == "__main__":
    main()

"""Round 41: PIT latest earnings-to-price characteristics for China A shares.

This reproduces the earnings-based value characteristic in Liu, Stambaugh
and Yuan's China model using the most recently disclosed non-recurring-item
adjusted profit and contemporaneous total-A-share market value.  A six-month
freshness version follows the paper's stale-accounting-information rule.

No label is read.  Negative earnings remain negative; missing or stale values
receive the same-date cross-sectional median so the full stock-day panel is
preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ch_factor_models import assign_available_trade_date, prepare_earnings


KEYS = ["TRADE_DATE", "SECURITY_ID"]
FACTOR_COLUMNS = ["r41_latest_cut_earnings_yield", "r41_fresh6m_cut_earnings_yield"]


def _files(root: Path, name: str) -> list[str]:
    files = sorted((root / name).glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(root / name)
    return [str(path) for path in files]


def _read(root: Path, name: str, columns: list[str]) -> pd.DataFrame:
    return ds.dataset(_files(root, name), format="parquet").to_table(columns=columns).to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = pd.to_numeric(panel["SECURITY_ID"]).astype("int64")
    calendar = pd.DatetimeIndex(panel["TRADE_DATE"].unique()).sort_values()

    earnings = _read(
        args.input_root,
        "earnings_pit",
        ["SECURITY_ID", "ID", "ACT_PUBTIME", "PUBLISH_DATE", "END_DATE", "N_INCOME_CUT"],
    )
    earnings = prepare_earnings(earnings, calendar, market_open="09:30:00")
    events = earnings[["SECURITY_ID", "AVAILABLE_DATE", "END_DATE", "N_INCOME_CUT"]].copy()
    events = events.sort_values(["AVAILABLE_DATE", "SECURITY_ID", "END_DATE"]).drop_duplicates(
        ["SECURITY_ID", "AVAILABLE_DATE"], keep="last"
    )

    market = _read(args.input_root, "market_daily", ["TRADE_DATE", "SECURITY_ID", "MARKET_VALUE_A"])
    market["TRADE_DATE"] = pd.to_datetime(market["TRADE_DATE"]).dt.normalize()
    market["SECURITY_ID"] = pd.to_numeric(market["SECURITY_ID"]).astype("int64")
    market = market.drop_duplicates(KEYS, keep="last")

    data = panel.merge(market, on=KEYS, how="left", validate="one_to_one")
    data = pd.merge_asof(
        data.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        events.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on="AVAILABLE_DATE",
        direction="backward",
    )
    denominator = pd.to_numeric(data["MARKET_VALUE_A"], errors="coerce")
    ep = pd.to_numeric(data["N_INCOME_CUT"], errors="coerce") / denominator
    ep = ep.where(denominator.gt(0)).replace([np.inf, -np.inf], np.nan).clip(-2.0, 2.0)
    age = (data["TRADE_DATE"] - pd.to_datetime(data["END_DATE"]).dt.normalize()).dt.days
    data["r41_latest_cut_earnings_yield"] = ep
    data["r41_fresh6m_cut_earnings_yield"] = ep.where(age.between(0, 183))
    medians = data.groupby("TRADE_DATE", sort=False)[FACTOR_COLUMNS].transform("median")
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].fillna(medians).fillna(0.0).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.sort_values(KEYS)[KEYS + FACTOR_COLUMNS].to_parquet(
        args.output, index=False, compression="zstd"
    )
    print(f"saved {len(data):,} rows to {args.output}")


if __name__ == "__main__":
    main()

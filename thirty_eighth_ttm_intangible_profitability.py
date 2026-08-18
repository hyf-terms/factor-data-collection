"""Round 38: TTM adaptation of literature intangible-adjusted profitability.

The annual Fama-French implementation is slow for the current 10-day label.
This PIT-safe adaptation keeps the published accounting adjustment (add back
R&D, optionally 30% SG&A) but applies it to TTM operating profitability.  A
cash-based operating-profit branch tests whether accrual removal complements
the same adjustment.  Coefficients are fixed ex ante; no labels are read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["TRADE_DATE", "SECURITY_ID"]
FACTOR_COLUMNS = [
    "r38_ttm_op_plus_rd_assets",
    "r38_ttm_op_plus_rd_30sga_assets",
    "r38_ttm_cbop_plus_rd_assets",
    "r38_ttm_cbop_plus_rd_30sga_assets",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--intangibles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    dense = pd.read_parquet(
        args.dense,
        columns=KEYS + ["dense_lit_ttm_op_assets", "dense_lit_ttm_cbop_assets"],
    )
    intangible = pd.read_parquet(
        args.intangibles,
        columns=KEYS + [
            "r36_rd_cap_investment_assets", "r36_org_cap_investment_assets",
        ],
    )
    for frame in [panel, dense, intangible]:
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
        frame["SECURITY_ID"] = frame["SECURITY_ID"].astype("int64")
    data = panel.merge(dense, on=KEYS, how="left", validate="one_to_one").merge(
        intangible, on=KEYS, how="left", validate="one_to_one"
    )
    inputs = [
        "dense_lit_ttm_op_assets", "dense_lit_ttm_cbop_assets",
        "r36_rd_cap_investment_assets", "r36_org_cap_investment_assets",
    ]
    data[inputs] = data[inputs].fillna(0.0)
    op = data["dense_lit_ttm_op_assets"]
    cbop = data["dense_lit_ttm_cbop_assets"]
    rd = data["r36_rd_cap_investment_assets"]
    sga = data["r36_org_cap_investment_assets"]
    data["r38_ttm_op_plus_rd_assets"] = op + rd
    data["r38_ttm_op_plus_rd_30sga_assets"] = op + rd + 0.30 * sga
    data["r38_ttm_cbop_plus_rd_assets"] = cbop + rd
    data["r38_ttm_cbop_plus_rd_30sga_assets"] = cbop + rd + 0.30 * sga
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].clip(-20, 20).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data[KEYS + FACTOR_COLUMNS].to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(data):,} rows")


if __name__ == "__main__":
    main()

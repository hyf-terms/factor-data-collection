"""Round 37: Jagannathan-Korajczyk-Wang intangible-adjusted profitability.

The paper modifies Fama-French operating profitability by adding back (a)
reported R&D or (b) R&D plus 30% of SG&A.  This module applies those published
accounting identities directly to PIT annual components; it performs no rank,
label fitting, or free-weight search.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["TRADE_DATE", "SECURITY_ID"]
FACTOR_COLUMNS = [
    "r37_ff_profit_plus_rd",
    "r37_ff_profit_plus_rd_30sga",
    "r37_ff_profit_plus_rd_fullsga",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--intangibles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    ff = pd.read_parquet(
        args.dense, columns=KEYS + ["dense_lit_ff_op_book_equity_annual"]
    )
    intangible = pd.read_parquet(
        args.intangibles,
        columns=KEYS + [
            "r36_rd_exp_book_equity", "r36_org_cap_investment_book_equity",
        ],
    )
    for frame in [base, ff, intangible]:
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
        frame["SECURITY_ID"] = frame["SECURITY_ID"].astype("int64")
    data = base.merge(ff, on=KEYS, how="left", validate="one_to_one").merge(
        intangible, on=KEYS, how="left", validate="one_to_one"
    )
    columns = [
        "dense_lit_ff_op_book_equity_annual", "r36_rd_exp_book_equity",
        "r36_org_cap_investment_book_equity",
    ]
    data[columns] = data[columns].fillna(0.0)
    ff_profit = data["dense_lit_ff_op_book_equity_annual"]
    rd = data["r36_rd_exp_book_equity"]
    sga = data["r36_org_cap_investment_book_equity"]
    data["r37_ff_profit_plus_rd"] = ff_profit + rd
    data["r37_ff_profit_plus_rd_30sga"] = ff_profit + rd + 0.30 * sga
    # The paper reports full SG&A as a robustness version; retain it as the
    # third and final pre-specified formula, not a searched weight.
    data["r37_ff_profit_plus_rd_fullsga"] = ff_profit + rd + sga
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].clip(-20, 20).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data[KEYS + FACTOR_COLUMNS].to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(data):,} rows")


if __name__ == "__main__":
    main()

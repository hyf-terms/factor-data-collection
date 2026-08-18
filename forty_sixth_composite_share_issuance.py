"""Round 46: Composite Share Issuance (CSI) for China A shares.

CSI over h trading observations is log market-cap growth minus log adjusted
price growth.  It isolates the growth in effective shares outstanding caused
by issuance, repurchases and share-count changes.  The factor is signed as
low issuance (negative CSI), consistent with the financing-anomaly literature.

Fixed 1-, 2- and 3-trading-year horizons are emitted.  No label is read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


KEYS = ["TRADE_DATE", "SECURITY_ID"]
HORIZONS = {"r46_low_csi_1y": 252, "r46_low_csi_2y": 504, "r46_low_csi_3y": 756}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.market_root.glob("**/*.parquet"))
    market = ds.dataset([str(p) for p in files], format="parquet").to_table(
        columns=KEYS + ["MARKET_VALUE_A", "ADJ_CLOSE_PRICE"]
    ).to_pandas()
    market["TRADE_DATE"] = pd.to_datetime(market["TRADE_DATE"]).dt.normalize()
    market["SECURITY_ID"] = pd.to_numeric(market["SECURITY_ID"]).astype("int64")
    market = market.sort_values(["SECURITY_ID", "TRADE_DATE"]).drop_duplicates(KEYS, keep="last")
    mv = pd.to_numeric(market["MARKET_VALUE_A"], errors="coerce").where(lambda x: x.gt(0))
    px = pd.to_numeric(market["ADJ_CLOSE_PRICE"], errors="coerce").where(lambda x: x.gt(0))
    for factor, lag in HORIZONS.items():
        lag_mv = mv.groupby(market["SECURITY_ID"], sort=False).shift(lag)
        lag_px = px.groupby(market["SECURITY_ID"], sort=False).shift(lag)
        csi = np.log(mv / lag_mv) - np.log(px / lag_px)
        market[factor] = (-csi).replace([np.inf, -np.inf], np.nan).clip(-5, 5).astype("float32")
    signal = market[KEYS + list(HORIZONS)].copy()
    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = pd.to_numeric(panel["SECURITY_ID"]).astype("int64")
    data = panel.merge(signal, on=KEYS, how="left", validate="one_to_one")
    coverage = []
    for year, group in data.groupby(data["TRADE_DATE"].dt.year, sort=True):
        for factor in HORIZONS:
            coverage.append({"year": int(year), "factor": factor, "missing_rate_before_fill": float(group[factor].isna().mean())})
    # A lag unavailable for young listings is a neutral signal, not a reason
    # to remove the stock. Same-day medians retain the complete test universe.
    medians = data.groupby("TRADE_DATE", sort=False)[list(HORIZONS)].transform("median")
    data[list(HORIZONS)] = data[list(HORIZONS)].fillna(medians).fillna(0.0).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.sort_values(KEYS)[KEYS + list(HORIZONS)].to_parquet(args.output, index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(args.coverage, index=False, encoding="utf-8-sig")
    print(f"saved {len(data):,} rows")


if __name__ == "__main__":
    main()

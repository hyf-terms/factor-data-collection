"""Build a compact raw-factor pool for IC10-series correlation testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEYS = ["TRADE_DATE", "SECURITY_ID"]
SOURCES = {
    "round14": (
        Path("artifacts/round14/anchor_increment/round14_anchor_increment_candidates.parquet"),
        ["r14_anchor_baseline"],
    ),
    "round19": (
        Path("artifacts/round19/new_effective_factors_round18_19.parquet"),
        [
            "r18_profitability_state_equal4",
            "r19_profit_confirmation",
            "r19_multi_horizon_profit",
        ],
    ),
    "round7": (
        Path(r"C:\Users\hyf\Desktop\因子\新测试结果\第七轮文献稠密财务因子\round7_selected.parquet"),
        ["r7_anchor_qop_incremental_w20", "r7_anchor_qprofit_incremental_w20"],
    ),
    "round43": (
        Path("artifacts/round43/round43_profitability_persistence_confirmation.parquet"),
        ["r43_profitability_persistence_confirmation_equal6"],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_columns = [column for _, columns in SOURCES.values() for column in columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for year in range(2017, 2027):
            filters = [
                ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
            ]
            data = pd.read_parquet(args.panel, columns=KEYS, filters=filters).drop_duplicates(KEYS)
            data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"]).dt.normalize()
            data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"]).astype("int64")
            for path, columns in SOURCES.values():
                source = pd.read_parquet(path, columns=KEYS + columns, filters=filters).drop_duplicates(KEYS)
                source["TRADE_DATE"] = pd.to_datetime(source["TRADE_DATE"]).dt.normalize()
                source["SECURITY_ID"] = pd.to_numeric(source["SECURITY_ID"]).astype("int64")
                data = data.merge(source, on=KEYS, how="left", validate="one_to_one")
            # PIT state values may end a few days before the current panel.
            data = data.sort_values(["SECURITY_ID", "TRADE_DATE"])
            data[all_columns] = data.groupby("SECURITY_ID", sort=False)[all_columns].ffill()
            medians = data.groupby("TRADE_DATE", sort=False)[all_columns].transform("median")
            data[all_columns] = data[all_columns].fillna(medians).fillna(0.0).astype("float32")
            output = data.sort_values(KEYS)[KEYS + all_columns]
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(output)
            print(year, f"rows={len(output):,}", "missing=0")
    finally:
        if writer is not None:
            writer.close()
    print(f"saved {total:,} rows and {len(all_columns)} factors to {args.output}")


if __name__ == "__main__":
    main()

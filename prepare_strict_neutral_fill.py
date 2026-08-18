"""Create full-panel neutral-fill versions of low-missing factor columns."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEYS = ["TRADE_DATE", "SECURITY_ID"]


def fill_daily_neutral(frame: pd.DataFrame, factor: str) -> tuple[pd.Series, int]:
    """Fill missing observations with that date's cross-sectional median."""
    values = pd.to_numeric(frame[factor], errors="coerce")
    medians = values.groupby(frame["TRADE_DATE"], sort=False).transform("median")
    if medians.loc[values.isna()].isna().any():
        raise ValueError(f"{factor} 存在整日全空，不能做中性值填充")
    missing = int(values.isna().sum())
    return values.fillna(medians), missing


def build(
    panel_path: Path,
    source_path: Path,
    factor: str,
    output_path: Path,
) -> None:
    output_factor = f"{factor}_neutral_fill"
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    years = sorted(pd.to_datetime(dates).dt.year.unique())
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    report = []
    try:
        for year in years:
            filters = [
                ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
            ]
            panel = pd.read_parquet(panel_path, columns=KEYS, filters=filters)
            source = pd.read_parquet(
                source_path, columns=KEYS + [factor], filters=filters
            )
            panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
            source["TRADE_DATE"] = pd.to_datetime(source["TRADE_DATE"]).dt.normalize()
            if panel.duplicated(KEYS).any() or source.duplicated(KEYS).any():
                raise ValueError(f"{year}存在重复键")
            merged = panel.merge(source, on=KEYS, how="left", validate="one_to_one")
            filled, missing = fill_daily_neutral(merged, factor)
            result = merged[KEYS].copy()
            result[output_factor] = filled.astype("float32")
            varying = result.groupby("TRADE_DATE")[output_factor].nunique().ge(2)
            if not varying.all():
                bad = varying.index[~varying].strftime("%Y-%m-%d").tolist()[:5]
                raise ValueError(f"填充后仍有恒定日: {bad}")
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            report.append(
                {
                    "year": year,
                    "rows": len(result),
                    "neutral_filled_rows": missing,
                    "neutral_filled_ratio": missing / len(result),
                    "constant_days_after_fill": int((~varying).sum()),
                }
            )
            print(f"{year}: rows={len(result):,}, filled={missing:,}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成数据")
    os.replace(temporary, output_path)
    pd.DataFrame(report).to_csv(
        output_path.with_suffix(".fill_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.panel.resolve(),
        args.source.resolve(),
        args.factor,
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

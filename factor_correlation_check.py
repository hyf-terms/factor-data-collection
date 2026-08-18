"""Measure daily cross-sectional correlations among tested factor residuals."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEYS = ["TRADE_DATE", "SECURITY_ID"]


def _available_factor_columns(path: Path) -> list[str]:
    return [column for column in pq.read_schema(path).names if column not in KEYS]


def load_factors(paths: list[Path], selected: list[str] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    requested = set(selected or [])
    for path in paths:
        available = _available_factor_columns(path)
        columns = [column for column in available if not requested or column in requested]
        duplicates = sorted(seen.intersection(columns))
        if duplicates:
            raise ValueError(f"duplicate factor names across inputs: {duplicates}")
        if not columns:
            continue
        frame = pd.read_parquet(path, columns=KEYS + columns)
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"], errors="coerce").dt.normalize()
        frames.append(frame.dropna(subset=KEYS))
        seen.update(columns)
    missing = sorted(requested.difference(seen))
    if missing:
        raise KeyError(f"requested factors not found: {missing}")
    if len(seen) < 2:
        raise ValueError("at least two factor columns are required")
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=KEYS, how="inner", validate="one_to_one")
    return result.sort_values(KEYS)


def daily_spearman(
    data: pd.DataFrame, factors: list[str], min_cross_section: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, group in data.groupby("TRADE_DATE", sort=True):
        for factor_a, factor_b in combinations(factors, 2):
            pair = group[[factor_a, factor_b]].dropna()
            n = len(pair)
            if n < min_cross_section:
                rho = np.nan
            else:
                ranked = pair.rank(method="average")
                rho = ranked[factor_a].corr(ranked[factor_b])
            rows.append(
                {
                    "TRADE_DATE": trade_date,
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "spearman_rho": rho,
                    "cross_section_n": n,
                }
            )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    valid = daily.dropna(subset=["spearman_rho"]).copy()
    valid["abs_rho"] = valid["spearman_rho"].abs()
    grouped = valid.groupby(["factor_a", "factor_b"], sort=True)
    summary = grouped.agg(
        mean_rho=("spearman_rho", "mean"),
        median_rho=("spearman_rho", "median"),
        mean_abs_rho=("abs_rho", "mean"),
        max_abs_rho=("abs_rho", "max"),
        effective_days=("spearman_rho", "count"),
        average_cross_section=("cross_section_n", "mean"),
    ).reset_index()
    p90 = grouped["abs_rho"].quantile(0.90).rename("p90_abs_rho").reset_index()
    return summary.merge(p90, on=["factor_a", "factor_b"], how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--factor-columns", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--min-cross-section", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_factors([path.resolve() for path in args.inputs], args.factor_columns)
    if args.start_date:
        data = data.loc[data["TRADE_DATE"].ge(pd.Timestamp(args.start_date))]
    if args.end_date:
        data = data.loc[data["TRADE_DATE"].le(pd.Timestamp(args.end_date))]
    factors = [column for column in data.columns if column not in KEYS]
    daily = daily_spearman(data, factors, args.min_cross_section)
    summary = summarize(daily)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output / "daily_factor_correlations.parquet", index=False, compression="zstd")
    summary.to_csv(output / "factor_correlation_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

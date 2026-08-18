"""Summarize strict factor eligibility and IC results in a separate folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csvs(root: Path, filename: str) -> list[tuple[str, pd.DataFrame]]:
    frames = []
    for path in sorted(root.rglob(filename)):
        source = path.parent.relative_to(root).as_posix()
        frames.append((source, pd.read_csv(path)))
    return frames


def summarize(base_dir: Path, result_root: Path) -> None:
    old = pd.read_csv(base_dir / "有效因子" / "因子分类清单.csv")
    old["abs_mean_ic"] = pd.to_numeric(old["abs_mean_ic"], errors="coerce")
    old = (
        old.sort_values("abs_mean_ic", ascending=False)
        .drop_duplicates("factor")
        [["factor", "mean_ic", "abs_mean_ic", "classification"]]
        .rename(
            columns={
                "mean_ic": "historical_mean_ic",
                "abs_mean_ic": "historical_abs_ic",
                "classification": "historical_classification",
            }
        )
    )

    quality_parts = []
    for source, frame in _read_csvs(result_root, "strict_factor_eligibility.csv"):
        frame.insert(0, "source_batch", source)
        quality_parts.append(frame)
    if not quality_parts:
        raise FileNotFoundError(f"{result_root} 下没有严格预检结果")
    quality = pd.concat(quality_parts, ignore_index=True)
    numeric = [
        "reference_rows",
        "reference_days",
        "missing_rows",
        "incomplete_days",
        "all_nan_days",
        "single_value_days",
        "low_variance_days",
    ]
    for column in numeric:
        quality[column] = pd.to_numeric(quality[column], errors="coerce").fillna(0)
    quality["quant_usable"] = (
        quality["quant_usable"].astype(str).str.lower().eq("true")
    )
    grouped = quality.groupby(["source_batch", "factor"], as_index=False)
    eligibility = grouped.agg(
        reference_rows=("reference_rows", "sum"),
        reference_days=("reference_days", "sum"),
        missing_rows=("missing_rows", "sum"),
        incomplete_days=("incomplete_days", "sum"),
        all_nan_days=("all_nan_days", "sum"),
        single_value_days=("single_value_days", "sum"),
        low_variance_days=("low_variance_days", "sum"),
        strict_pass=("quant_usable", "all"),
    )
    eligibility["missing_ratio"] = np.where(
        eligibility["reference_rows"].gt(0),
        eligibility["missing_rows"] / eligibility["reference_rows"],
        np.nan,
    )
    eligibility = eligibility.merge(old, on="factor", how="left")
    eligibility["low_missing_candidate"] = (
        eligibility["missing_ratio"].le(0.10)
        & eligibility["historical_abs_ic"].ge(0.025)
    )
    eligibility = eligibility.sort_values(
        ["strict_pass", "low_missing_candidate", "historical_abs_ic"],
        ascending=[False, False, False],
    )
    eligibility.to_csv(
        result_root / "严格预检汇总.csv", index=False, encoding="utf-8-sig"
    )
    eligibility.loc[eligibility["low_missing_candidate"]].to_csv(
        result_root / "低空值候选.csv", index=False, encoding="utf-8-sig"
    )

    ic_parts = []
    for source, frame in _read_csvs(result_root, "ic_summary.csv"):
        frame = frame.loc[frame["version"].eq("neutral")].copy()
        frame.insert(0, "source_batch", source)
        ic_parts.append(frame)
    ic = pd.concat(ic_parts, ignore_index=True) if ic_parts else pd.DataFrame()
    ic.to_csv(result_root / "严格IC汇总.csv", index=False, encoding="utf-8-sig")

    print(f"strict_pass={int(eligibility.strict_pass.sum())}")
    print(f"low_missing_candidates={int(eligibility.low_missing_candidate.sum())}")
    if not ic.empty:
        print(
            ic[["factor", "mean_ic", "effective_days"]]
            .sort_values("mean_ic", ascending=False)
            .to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    result_root = (
        args.result_root.resolve()
        if args.result_root
        else base_dir / "新测试结果"
    )
    summarize(base_dir, result_root)


if __name__ == "__main__":
    main()

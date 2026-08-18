"""Greedy selection of high-IC factors with low daily-IC correlation.

Inputs are one or more ``daily_ic.csv/parquet`` files produced by
``factors_neus_only2.py``.  Only the neutralized version is used when a
``version`` column is present.  Candidates first pass the absolute mean-IC
threshold, then are visited by descending absolute mean IC.  A candidate is
kept only when its absolute Pearson correlation with every selected daily-IC
series is strictly below the configured threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATE = "TRADE_DATE"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def load_daily_ic(paths: list[Path], version: str = "neutral") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        data = _read(path)
        if DATE not in data:
            raise KeyError(f"{path} missing {DATE}")
        data[DATE] = pd.to_datetime(data[DATE], errors="coerce").dt.normalize()
        if "version" in data:
            data = data.loc[data["version"].astype(str).eq(version)]
        version_ic = f"{version}_ic"
        if {"factor", version_ic}.issubset(data.columns):
            long = data[[DATE, "factor", version_ic]].rename(
                columns={version_ic: "ic"}
            )
        elif {"factor", "ic"}.issubset(data.columns):
            long = data[[DATE, "factor", "ic"]].copy()
        else:
            value_columns = [
                column for column in data.columns
                if column not in {DATE, "version", "n", "cross_section_n"}
            ]
            if not value_columns:
                raise ValueError(f"{path} has no IC value columns")
            long = data.melt(
                id_vars=[DATE], value_vars=value_columns,
                var_name="factor", value_name="ic",
            )
        long["ic"] = pd.to_numeric(long["ic"], errors="coerce")
        frames.append(long.dropna(subset=[DATE, "factor", "ic"]))
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated([DATE, "factor"], keep=False)
    if duplicates.any():
        examples = combined.loc[duplicates, [DATE, "factor"]].head().to_dict("records")
        raise ValueError(f"duplicate daily IC observations: {examples}")
    return combined


def greedy_select(
    daily: pd.DataFrame,
    ic_threshold: float = 0.035,
    correlation_threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = daily.groupby("factor", sort=True)["ic"].mean()
    candidates = means[means.abs().ge(ic_threshold)].sort_values(
        key=lambda values: values.abs(), ascending=False
    )
    if candidates.empty:
        raise ValueError("no factor passes the absolute mean-IC threshold")
    wide = daily.loc[daily.factor.isin(candidates.index)].pivot(
        index=DATE, columns="factor", values="ic"
    )
    correlation = wide.corr(method="pearson", min_periods=30).abs()

    selected: list[str] = []
    decisions: list[dict[str, object]] = []
    for factor, mean_ic in candidates.items():
        correlations = correlation.loc[factor, selected] if selected else pd.Series(dtype=float)
        max_corr = float(correlations.max()) if not correlations.empty else 0.0
        blocker = str(correlations.idxmax()) if not correlations.empty else ""
        keep = not selected or (correlations.notna().all() and correlations.lt(correlation_threshold).all())
        if keep:
            selected.append(factor)
        decisions.append({
            "factor": factor,
            "mean_ic": float(mean_ic),
            "abs_mean_ic": float(abs(mean_ic)),
            "selected": bool(keep),
            "max_abs_corr_to_earlier_selected": max_corr,
            "blocking_factor": "" if keep else blocker,
        })
    return pd.DataFrame(decisions), correlation, wide[selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-ic", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="neutral")
    parser.add_argument("--ic-threshold", type=float, default=0.035)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    args = parser.parse_args()
    daily = load_daily_ic([path.resolve() for path in args.daily_ic], args.version)
    decisions, correlation, selected_daily = greedy_select(
        daily, args.ic_threshold, args.correlation_threshold
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output / "greedy_selection.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(output / "daily_ic_abs_correlation.csv", encoding="utf-8-sig")
    selected_daily.to_parquet(output / "selected_daily_ic.parquet", compression="zstd")
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()

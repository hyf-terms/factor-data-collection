"""Dense all-quarter optimization of strong legacy factor families.

The new variants use the latest disclosed Q1-Q4 information until replaced,
have no fixed 60-day expiry, and use robust Z scores instead of percentile
ranks.  Six previously built dense variables are included unchanged in the
same output so the strict tester evaluates every column on one common panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dense_no_rank_factor_optimization import (
    RAW_SIGNALS as DENSE_RAW_SIGNALS,
    SUE_SIGNALS,
    map_signal_inputs,
    prepare_raw_events,
    prepare_sue_groups,
)
from all_quarter_raw_factor_search import (
    SIGNAL_COLUMNS as FULL_RAW_SIGNALS,
    attach_prior_year,
    calculate_event_signals,
    load_all_quarter_events,
    prepare_available_events,
)
from coverage_literature_factor_search import map_events_to_panel
from dense_q1_gross_profit_factors import robust_daily_zscore
from event_financial_factor_search import KEYS, _normalize_panel
from third_round_new_factor_optimization import _daily_neutral_fill


BASE_DIR = Path(__file__).resolve().parent
LEGACY_COMPONENTS = [
    "quarterly_f_score",
    "operating_profit_acceleration",
    "cfo_sue",
    "accrual_quality",
    "asset_growth",
    "profitability_quality_score",
    "gross_profitability",
]
EXISTING_SIX = [
    "dense_q1_profit_growth",
    "dense_q1_growth_cash_breadth",
    "dense_q1_margin_cash_improvement",
    "dense_q1_rd_growth_efficiency",
    "dense_q1_net_income_sue",
    "dense_q1_quality_growth_ensemble",
]
REJECTED_FINANCIAL = [
    "dense_allq_all_profit_surprise",
    "dense_allq_financial_composite",
    "dense_allq_financial_expanded",
    "dense_allq_financial_median",
    "dense_allq_financial_no_asset",
    "dense_allq_financial_no_quality",
    "dense_allq_financial_pead2",
]


def new_specs() -> list[tuple[str, str]]:
    return [
        ("dense_allq_financial_composite", "financial"),
        ("dense_allq_financial_pead2", "financial"),
        ("dense_allq_financial_median", "financial"),
        ("dense_allq_financial_no_quality", "financial"),
        ("dense_allq_financial_no_asset", "financial"),
        ("dense_allq_financial_expanded", "financial"),
        ("dense_allq_financial_cash_breadth", "cash_breadth"),
        ("dense_allq_cash_quality", "cash_breadth"),
        ("dense_allq_growth_consistency", "growth_consistency"),
        ("dense_allq_growth_consistency_direct", "growth_consistency"),
        ("dense_allq_cost_discipline", "cost_discipline"),
        ("dense_allq_cost_discipline_direct", "cost_discipline"),
        ("dense_allq_revenue_growth", "revenue_growth"),
        ("dense_allq_all_profit_surprise", "profit_cash"),
        ("dense_allq_joint_surprise_cash_quality", "profit_cash"),
        ("dense_allq_all_profit_cash_quality", "profit_cash"),
        ("dense_allq_all_profit_cash_penalty", "profit_cash"),
        ("dense_allq_surprise_growth_cash", "profit_cash"),
        ("dense_allq_profit_growth_consistency", "profit_cash"),
    ]


def candidate_names() -> list[str]:
    return [name for name, _ in new_specs()] + EXISTING_SIX


def _standardize(dates: pd.Series, values: pd.Series) -> pd.Series:
    return robust_daily_zscore(
        pd.DataFrame({"TRADE_DATE": dates, "value": values}), ["value"]
    )["value"].fillna(0.0)


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    source = [*FULL_RAW_SIGNALS, *SUE_SIGNALS, *LEGACY_COMPONENTS, *EXISTING_SIX]
    dense = _daily_neutral_fill(inputs, source)
    z = robust_daily_zscore(dense, source)
    dates = dense["TRADE_DATE"]

    financial_columns = [
        "NET_INCOME_SUE",
        "DEDUCTED_INCOME_SUE",
        "quarterly_f_score",
        "operating_profit_acceleration",
        "asset_growth",
        "profitability_quality_score",
    ]
    financial = z[financial_columns]
    financial_core = financial.mean(axis=1)
    financial_pead2 = (
        2 * z["NET_INCOME_SUE"]
        + 2 * z["DEDUCTED_INCOME_SUE"]
        + z["quarterly_f_score"]
        + z["operating_profit_acceleration"]
        + z["asset_growth"]
        + z["profitability_quality_score"]
    ) / 8
    financial_median = financial.median(axis=1)
    financial_no_quality = financial.drop(columns="profitability_quality_score").mean(axis=1)
    financial_no_asset = financial.drop(columns="asset_growth").mean(axis=1)
    financial_expanded = (
        financial_core * 6
        + z["GROSS_PROFIT_SUE"]
        + z["cfo_sue"]
    ) / 8

    growth_columns = [
        "REVENUE_GROWTH", "PROFIT_GROWTH", "CFO_GROWTH", "SALES_CASH_GROWTH"
    ]
    growth_z = z[growth_columns]
    growth_breadth = growth_z.mean(axis=1)
    growth_consistency = growth_breadth - growth_z.std(axis=1, ddof=0)
    raw_growth = dense[growth_columns].clip(-5.0, 5.0)
    growth_consistency_direct = _standardize(
        dates, raw_growth.mean(axis=1) - raw_growth.std(axis=1, ddof=0)
    )
    cost_discipline = _standardize(
        dates,
        0.50 * (z["REVENUE_GROWTH"] - z["COGS_GROWTH"])
        + 0.50 * z["GROSS_PROFIT_GROWTH"],
    )
    cost_discipline_direct = _standardize(
        dates,
        0.50 * (dense["REVENUE_GROWTH"].clip(-5, 5) - dense["COGS_GROWTH"].clip(-5, 5))
        + 0.50 * dense["GROSS_PROFIT_GROWTH"].clip(-5, 5),
    )

    profit_surprise = z[SUE_SIGNALS].mean(axis=1)
    joint_surprise = (z["NET_INCOME_SUE"] + z["DEDUCTED_INCOME_SUE"]) / 2
    cash_quality = (
        z["DEDUCTED_INCOME_SUE"]
        + z["OPERATING_PROFIT_SUE"]
        + z["cfo_sue"]
        + z["accrual_quality"]
        + z["gross_profitability"]
    ) / 5
    financial_cash = _standardize(dates, 0.50 * financial_core + 0.50 * growth_breadth)
    cash_quality = _standardize(dates, cash_quality)
    penalty = profit_surprise + cash_quality.clip(upper=0.0)

    values = {
        "dense_allq_financial_composite": financial_core,
        "dense_allq_financial_pead2": financial_pead2,
        "dense_allq_financial_median": financial_median,
        "dense_allq_financial_no_quality": financial_no_quality,
        "dense_allq_financial_no_asset": financial_no_asset,
        "dense_allq_financial_expanded": financial_expanded,
        "dense_allq_financial_cash_breadth": financial_cash,
        "dense_allq_cash_quality": cash_quality,
        "dense_allq_growth_consistency": _standardize(dates, growth_consistency),
        "dense_allq_growth_consistency_direct": growth_consistency_direct,
        "dense_allq_cost_discipline": cost_discipline,
        "dense_allq_cost_discipline_direct": cost_discipline_direct,
        "dense_allq_revenue_growth": z["REVENUE_GROWTH"],
        "dense_allq_all_profit_surprise": _standardize(dates, profit_surprise),
        "dense_allq_joint_surprise_cash_quality": _standardize(
            dates, 0.60 * joint_surprise + 0.40 * cash_quality
        ),
        "dense_allq_all_profit_cash_quality": _standardize(
            dates, 0.60 * profit_surprise + 0.40 * cash_quality
        ),
        "dense_allq_all_profit_cash_penalty": _standardize(dates, penalty),
        "dense_allq_surprise_growth_cash": _standardize(
            dates, 0.50 * joint_surprise + 0.50 * growth_breadth
        ),
        "dense_allq_profit_growth_consistency": _standardize(
            dates, 0.60 * profit_surprise + 0.40 * growth_consistency
        ),
    }
    result = dense[KEYS].copy()
    for name, signal in values.items():
        result[name] = _standardize(dates, signal).astype("float32")
    for name in EXISTING_SIX:
        result[name] = pd.to_numeric(dense[name], errors="coerce").fillna(0).astype("float32")
    return result[KEYS + candidate_names()]


def generate(panel_path: Path, data_root: Path, dense_path: Path, output: Path, manifest: Path, quality: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    ordinal = pd.Series(np.arange(len(calendar), dtype=np.int16), index=calendar)
    raw_events = prepare_raw_events(data_root, calendar)
    full_raw_events = prepare_available_events(
        calculate_event_signals(attach_prior_year(load_all_quarter_events(data_root))),
        calendar,
    )
    sue_groups = prepare_sue_groups(BASE_DIR, calendar)
    specs = pd.DataFrame(new_specs(), columns=["factor", "family"])
    existing = pd.DataFrame({"factor": EXISTING_SIX, "family": "existing_six"})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([specs, existing], ignore_index=True).to_csv(manifest, index=False, encoding="utf-8-sig")
    temporary = output.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    reports = []
    try:
        for year in sorted(set(calendar.year)):
            filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
            panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS + LEGACY_COMPONENTS, filters=filters))
            signals = map_signal_inputs(panel[KEYS], raw_events, sue_groups, ordinal)
            raw = map_events_to_panel(panel[KEYS], full_raw_events, ordinal)
            previous = pd.read_parquet(dense_path, columns=KEYS + EXISTING_SIX, filters=filters)
            inputs = panel.merge(raw[KEYS + FULL_RAW_SIGNALS], on=KEYS, how="left", validate="one_to_one")
            inputs = inputs.merge(signals[KEYS + SUE_SIGNALS], on=KEYS, how="left", validate="one_to_one")
            inputs = inputs.merge(previous, on=KEYS, how="left", validate="one_to_one")
            candidates = build_candidates(inputs)
            for factor in candidate_names():
                grouped = candidates.groupby("TRADE_DATE")[factor]
                reports.append({
                    "year": year, "factor": factor,
                    "missing_rows": int(candidates[factor].isna().sum()),
                    "constant_days": int(grouped.nunique(dropna=True).lt(2).sum()),
                })
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            print(f"{year}: rows={len(candidates):,}, factors={len(candidate_names())}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成候选")
    temporary.replace(output)
    pd.DataFrame(reports).to_csv(quality, index=False, encoding="utf-8-sig")


def summarize(daily_ic_path: Path, manifest_path: Path, report: Path, extra_daily_ic: Path | None = None) -> None:
    frames = [pd.read_parquet(daily_ic_path)]
    if extra_daily_ic is not None and extra_daily_ic.exists():
        frames.append(pd.read_parquet(extra_daily_ic))
    data = pd.concat(frames, ignore_index=True)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"])
    manifest = pd.read_csv(manifest_path)
    rows = []
    for factor, group in data.groupby("factor"):
        train = group.loc[group["TRADE_DATE"].dt.year.between(2017, 2022)]
        yearly = train.groupby(train["TRADE_DATE"].dt.year)["neutral_ic"].mean()
        rows.append({
            "factor": factor,
            "full_ic": group["neutral_ic"].mean(),
            "train_2017_2022_ic": train["neutral_ic"].mean(),
            "validation_2023_2024_ic": group.loc[group["TRADE_DATE"].dt.year.between(2023, 2024), "neutral_ic"].mean(),
            "holdout_2025_2026_ic": group.loc[group["TRADE_DATE"].dt.year.between(2025, 2026), "neutral_ic"].mean(),
            "positive_train_years": int(yearly.gt(0).sum()),
        })
    result = pd.DataFrame(rows)
    result["base_factor"] = result["factor"].str.removesuffix("_anchor10")
    result = result.merge(
        manifest.rename(columns={"factor": "base_factor"}), on="base_factor", how="left"
    )
    result["selected_in_family"] = False
    for family, positions in result.groupby("family").groups.items():
        subset = result.loc[positions]
        eligible = subset.loc[subset["positive_train_years"].ge(5)]
        pool = eligible if not eligible.empty else subset
        selected = pool["train_2017_2022_ic"].idxmax()
        result.loc[selected, "selected_in_family"] = True
    result.sort_values("full_ic", ascending=False).to_csv(report, index=False, encoding="utf-8-sig")


def repair_financial_candidates(grid_path: Path, output: Path) -> None:
    """Use a 10% dense earnings anchor on otherwise all-neutral early days."""
    columns = KEYS + REJECTED_FINANCIAL + ["dense_q1_net_income_sue"]
    data = pd.read_parquet(grid_path, columns=columns)
    result = data[KEYS].copy()
    anchor = pd.to_numeric(data["dense_q1_net_income_sue"], errors="coerce").fillna(0.0)
    for factor in REJECTED_FINANCIAL:
        result[f"{factor}_anchor10"] = (
            0.90 * pd.to_numeric(data[factor], errors="coerce").fillna(0.0)
            + 0.10 * anchor
        ).astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")


def finalize_selected(grid_path: Path, repaired_path: Path, output: Path) -> None:
    """Export training-selected representatives plus the six requested variables."""
    grid_selected = [
        "dense_allq_financial_cash_breadth",
        "dense_allq_growth_consistency",
        "dense_allq_cost_discipline",
        "dense_allq_revenue_growth",
        "dense_allq_surprise_growth_cash",
        *EXISTING_SIX,
    ]
    repaired_selected = ["dense_allq_financial_no_asset_anchor10"]
    grid = pd.read_parquet(grid_path, columns=KEYS + grid_selected)
    repaired = pd.read_parquet(repaired_path, columns=KEYS + repaired_selected)
    result = grid.merge(repaired, on=KEYS, how="inner", validate="one_to_one")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")


def main() -> None:
    root = BASE_DIR / "新测试结果" / "第五轮旧高IC族稠密优化"
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data" / "new_pit")
    parser.add_argument("--dense", type=Path, default=BASE_DIR / "factor_components" / "dense_no_rank_factors.parquet")
    parser.add_argument("--output", type=Path, default=root / "round5_grid_with_existing_six.parquet")
    parser.add_argument("--manifest", type=Path, default=root / "round5_manifest.csv")
    parser.add_argument("--quality", type=Path, default=root / "round5_quality.csv")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--daily-ic", type=Path, default=root / "strict_test" / "daily_ic.parquet")
    parser.add_argument("--report", type=Path, default=root / "round5_selection_report.csv")
    parser.add_argument("--repair-financial", action="store_true")
    parser.add_argument("--repair-output", type=Path, default=root / "round5_repaired_financial.parquet")
    parser.add_argument("--extra-daily-ic", type=Path, default=root / "repaired_test" / "daily_ic.parquet")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--selected-output", type=Path, default=root / "round5_selected_with_existing_six.parquet")
    args = parser.parse_args()
    if args.finalize:
        finalize_selected(args.output.resolve(), args.repair_output.resolve(), args.selected_output.resolve())
    elif args.repair_financial:
        repair_financial_candidates(args.output.resolve(), args.repair_output.resolve())
    elif args.summarize:
        summarize(
            args.daily_ic.resolve(), args.manifest.resolve(), args.report.resolve(),
            args.extra_daily_ic.resolve(),
        )
    else:
        generate(args.panel.resolve(), args.data_root.resolve(), args.dense.resolve(), args.output.resolve(), args.manifest.resolve(), args.quality.resolve())


if __name__ == "__main__":
    main()

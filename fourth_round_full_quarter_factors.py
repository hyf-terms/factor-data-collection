"""Full-quarter, no-expiry tests for four low-correlation financial themes.

The newest disclosed Q1/Q2/Q3/Q4 observation remains active until replaced.
There is no Q1 filter and no 60-day expiry.  Inputs are daily robust-Z
standardized and never percentile-ranked.  Incremental variants are daily
cross-sectional residuals against the current best profitability factor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from coverage_literature_factor_search import map_events_to_panel
from dense_q1_gross_profit_factors import robust_daily_zscore
from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric
from raw_q1_minimal_factor_search import _safe_divide
from third_round_new_factor_optimization import (
    _daily_neutral_fill,
    _daily_orthogonal_residual,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_FIELDS = ["REVENUE", "SELL_EXP", "ADMIN_EXP", "OPERATE_PROFIT", "COGS"]
RAW_SIGNALS = [
    "RAW_REVENUE_GROWTH", "RAW_SGA_GROWTH", "RAW_OPER_PROFIT_GROWTH",
    "RAW_GROSS_PROFIT_GROWTH",
]
VENDOR_FIELDS = [
    "REVENUE_YOY", "REVENUE_QOQ", "OPER_PROFIT_YOY", "OPER_PROFIT_QOQ",
    "GROSS_PROFIT_YOY", "GROSS_PROFIT_QOQ", "N_CF_OPA_YOY", "N_CF_OPA_QOQ",
    "C_FR_SALE_G_S_YOY", "C_FR_SALE_G_S_QOQ", "CFSGS_R",
]
PANEL_FIELDS = ["cfo_sue"]
ANCHOR = "optimized_interaction"


def candidate_names() -> list[str]:
    components = [
        "r4_cost_stickiness_gap",
        "r4_cost_stickiness_downturn",
        "r4_operating_leverage_yoy",
        "r4_operating_leverage_qoq",
        "r4_cfo_surprise_ensemble",
        "r4_cfo_yoy_surprise",
        "r4_sales_collection_yoy_match",
        "r4_sales_collection_qoq_match",
    ]
    groups = ["cost_stickiness", "operating_leverage", "cfo_surprise", "sales_collection"]
    return components + [f"r4_{name}_composite" for name in groups] + [
        f"r4_{name}_incremental" for name in groups
    ]


def _load_raw_income(data_root: Path) -> pd.DataFrame:
    source = pd.read_parquet(data_root / "new_pit_income", columns=[*COMMON_COLUMNS, *RAW_FIELDS])
    frames = {
        field: build_standalone_quarterly_metric(source, field, name=f"r4:{field}")
        for field in RAW_FIELDS
    }
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    base = frames["REVENUE"][keys + ["FISCAL_QUARTER", "EVENT_TIME", "REVENUE"]].rename(
        columns={"EVENT_TIME": "REVENUE_EVENT_TIME"}
    )
    for field in RAW_FIELDS[1:]:
        metric = frames[field][keys + ["EVENT_TIME", field]].rename(
            columns={"EVENT_TIME": f"{field}_EVENT_TIME"}
        )
        base = base.merge(metric, on=keys, how="left", validate="one_to_one")
    base["EVENT_TIME"] = base[[f"{field}_EVENT_TIME" for field in RAW_FIELDS]].max(axis=1)
    prior = base[["SECURITY_ID", "QUARTER_INDEX", *RAW_FIELDS]].copy()
    prior["QUARTER_INDEX"] += 4
    prior = prior.rename(columns={field: f"PRIOR_{field}" for field in RAW_FIELDS})
    data = base.merge(prior, on=["SECURITY_ID", "QUARTER_INDEX"], how="left", validate="one_to_one")
    current_sga = data["SELL_EXP"].fillna(0) + data["ADMIN_EXP"].fillna(0)
    prior_sga = data["PRIOR_SELL_EXP"].fillna(0) + data["PRIOR_ADMIN_EXP"].fillna(0)
    current_gp = data["REVENUE"] - data["COGS"]
    prior_gp = data["PRIOR_REVENUE"] - data["PRIOR_COGS"]
    specs = {
        "RAW_REVENUE_GROWTH": (data["REVENUE"], data["PRIOR_REVENUE"]),
        "RAW_SGA_GROWTH": (current_sga, prior_sga),
        "RAW_OPER_PROFIT_GROWTH": (data["OPERATE_PROFIT"], data["PRIOR_OPERATE_PROFIT"]),
        "RAW_GROSS_PROFIT_GROWTH": (current_gp, prior_gp),
    }
    for name, (current, prior_value) in specs.items():
        data[name] = _safe_divide(current - prior_value, prior_value, absolute_denominator=True)
    return data[["SECURITY_ID", "EVENT_TIME", "FISCAL_QUARTER", "QUARTER_INDEX", *RAW_SIGNALS]]


def _load_vendor_events(dataset: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID", "ID", "PUBLISH_DATE", "END_DATE_REP", "END_DATE",
        "UPDATE_TIME", "REPORT_TYPE", *VENDOR_FIELDS,
    ]
    data = pd.read_parquet(dataset, columns=columns)
    for column in ["PUBLISH_DATE", "END_DATE_REP", "END_DATE", "UPDATE_TIME"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = data.loc[
        data["END_DATE"].eq(data["END_DATE_REP"])
        & data["REPORT_TYPE"].isin(["Q1", "S1", "Q3", "A"])
    ].copy()
    quarter = {"Q1": 1, "S1": 2, "Q3": 3, "A": 4}
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(quarter)
    data["QUARTER_INDEX"] = data["END_DATE"].dt.year * 4 + data["FISCAL_QUARTER"]
    data["EVENT_TIME"] = data["PUBLISH_DATE"].dt.normalize() + pd.to_timedelta(86_399, unit="s")
    data = data.dropna(subset=["SECURITY_ID", "ID", "EVENT_TIME", "QUARTER_INDEX"])
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"]).astype("int64")
    data["ID"] = pd.to_numeric(data["ID"]).astype("int64")
    for field in VENDOR_FIELDS:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    return data.sort_values(
        ["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX", "UPDATE_TIME", "ID"]
    ).drop_duplicates(["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"], keep="last")


def _prepare_events(data: pd.DataFrame, calendar: pd.DatetimeIndex, fields: list[str]) -> pd.DataFrame:
    available = assign_available_trade_date(data, calendar).sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
    )
    newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
    available = available.loc[available["QUARTER_INDEX"].eq(newest)]
    return available.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")[
        ["SECURITY_ID", "AVAILABLE_DATE", "QUARTER_INDEX", *fields]
    ].reset_index(drop=True)


def _standardize(dates: pd.Series, values: pd.Series) -> pd.Series:
    return robust_daily_zscore(
        pd.DataFrame({"TRADE_DATE": dates, "value": values}), ["value"]
    )["value"].fillna(0.0)


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    fields = RAW_SIGNALS + VENDOR_FIELDS + PANEL_FIELDS + [ANCHOR]
    dense = _daily_neutral_fill(inputs, fields)
    z = robust_daily_zscore(dense, fields)
    dates = dense["TRADE_DATE"]

    raw_gap = z["RAW_REVENUE_GROWTH"] - z["RAW_SGA_GROWTH"]
    cost_gap = _standardize(dates, raw_gap)
    downturn = _standardize(
        dates, raw_gap * np.where(dense["RAW_REVENUE_GROWTH"].lt(0), 1.5, 1.0)
    )
    op_yoy = _standardize(dates, z["OPER_PROFIT_YOY"] - z["REVENUE_YOY"])
    op_qoq = _standardize(dates, z["OPER_PROFIT_QOQ"] - z["REVENUE_QOQ"])
    gross_yoy = _standardize(dates, z["GROSS_PROFIT_YOY"] - z["REVENUE_YOY"])
    gross_qoq = _standardize(dates, z["GROSS_PROFIT_QOQ"] - z["REVENUE_QOQ"])
    cfo_ensemble = _standardize(
        dates, 0.50 * z["cfo_sue"] + 0.30 * z["N_CF_OPA_YOY"] + 0.20 * z["N_CF_OPA_QOQ"]
    )
    cfo_yoy = z["N_CF_OPA_YOY"]
    collect_yoy = _standardize(dates, z["C_FR_SALE_G_S_YOY"] - z["REVENUE_YOY"])
    collect_qoq = _standardize(dates, z["C_FR_SALE_G_S_QOQ"] - z["REVENUE_QOQ"])

    components = {
        "cost_stickiness_gap": cost_gap,
        "cost_stickiness_downturn": downturn,
        "operating_leverage_yoy": op_yoy,
        "operating_leverage_qoq": op_qoq,
        "cfo_surprise_ensemble": cfo_ensemble,
        "cfo_yoy_surprise": cfo_yoy,
        "sales_collection_yoy_match": collect_yoy,
        "sales_collection_qoq_match": collect_qoq,
    }
    groups = {
        "cost_stickiness": _standardize(dates, 0.60 * cost_gap + 0.40 * downturn),
        "operating_leverage": _standardize(
            dates, 0.35 * op_yoy + 0.20 * op_qoq + 0.30 * gross_yoy + 0.15 * gross_qoq
        ),
        "cfo_surprise": cfo_ensemble,
        "sales_collection": _standardize(
            dates, 0.45 * collect_yoy + 0.30 * collect_qoq + 0.25 * z["CFSGS_R"]
        ),
    }
    result = dense[KEYS].copy()
    for name, values in components.items():
        result[f"r4_{name}"] = values
    anchor = z[ANCHOR]
    for name, values in groups.items():
        result[f"r4_{name}_composite"] = values
        result[f"r4_{name}_incremental"] = _daily_orthogonal_residual(values, anchor, dates)
    for factor in candidate_names():
        result[factor] = pd.to_numeric(result[factor], errors="coerce").fillna(0).astype("float32")
    return result


def generate(panel_path: Path, data_root: Path, indicator_path: Path, anchor_path: Path, output: Path, quality: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    ordinal = pd.Series(np.arange(len(calendar), dtype=np.int16), index=calendar)
    raw_events = _prepare_events(_load_raw_income(data_root), calendar, RAW_SIGNALS)
    vendor_events = _prepare_events(_load_vendor_events(indicator_path), calendar, VENDOR_FIELDS)
    names = candidate_names()
    temporary = output.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    reports: list[dict[str, object]] = []
    try:
        for year in sorted(set(calendar.year)):
            filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
            panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS + PANEL_FIELDS, filters=filters))
            raw = map_events_to_panel(panel[KEYS], raw_events, ordinal)
            vendor = map_events_to_panel(panel[KEYS], vendor_events, ordinal)
            anchor = pd.read_parquet(anchor_path, columns=KEYS + [ANCHOR], filters=filters)
            inputs = panel.merge(raw[KEYS + RAW_SIGNALS], on=KEYS, how="left", validate="one_to_one")
            inputs = inputs.merge(vendor[KEYS + VENDOR_FIELDS], on=KEYS, how="left", validate="one_to_one")
            inputs = inputs.merge(anchor, on=KEYS, how="left", validate="one_to_one")
            candidates = build_candidates(inputs)
            for factor in names:
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
            print(f"{year}: rows={len(candidates):,}, factors={len(names)}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成候选")
    temporary.replace(output)
    pd.DataFrame(reports).to_csv(quality, index=False, encoding="utf-8-sig")


def summarize_test(daily_ic_path: Path, output: Path) -> None:
    data = pd.read_parquet(daily_ic_path)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"])
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
            "all_train_years_positive": bool(len(yearly) == 6 and yearly.gt(0).all()),
            "eligible_incremental_blend": bool(
                factor.endswith("_incremental") and len(yearly) == 6 and yearly.gt(0).all()
            ),
        })
    pd.DataFrame(rows).sort_values("full_ic", ascending=False).to_csv(
        output, index=False, encoding="utf-8-sig"
    )


def main() -> None:
    root = BASE_DIR / "新测试结果" / "第四轮全季度"
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data" / "new_pit")
    parser.add_argument("--indicator-pit", type=Path, default=BASE_DIR / "data" / "quarterly_financial_indicators" / "quarterly_financial_indicator_pit")
    parser.add_argument("--anchor", type=Path, default=BASE_DIR / "新测试结果" / "第二轮优化" / "round2_selected.parquet")
    parser.add_argument("--output", type=Path, default=root / "round4_grid.parquet")
    parser.add_argument("--quality", type=Path, default=root / "round4_grid_quality.csv")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--daily-ic", type=Path, default=root / "grid_test" / "daily_ic.parquet")
    parser.add_argument("--report", type=Path, default=root / "round4_selection_report.csv")
    args = parser.parse_args()
    if args.summarize:
        summarize_test(args.daily_ic.resolve(), args.report.resolve())
    else:
        generate(args.panel.resolve(), args.data_root.resolve(), args.indicator_pit.resolve(), args.anchor.resolve(), args.output.resolve(), args.quality.resolve())


if __name__ == "__main__":
    main()

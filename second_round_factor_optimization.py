"""Second-round dense financial-factor optimization with standardized inputs."""

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
from first_round_factor_optimization import (
    _daily_median_fill,
    _industry_labels,
    _industry_robust_zscore,
    fit,
)
from factors_neus_only2 import INDUSTRY_FACTORS
from pead_sue_factor import assign_available_trade_date


BASE_DIR = Path(__file__).resolve().parent
FIELDS = [
    "GROSS_PROFIT_YOY",
    "NP_MARGIN_YOY",
    "P_COST_EXP",
    "PERIOD_EXP_TR",
    "AR_REC_R",
    "AR_R",
    "N_CF_OPA_YOY",
]
DENSE_FIELDS = [
    "dense_q1_earnings_sue_ensemble",
    "dense_q1_gross_profit_sue",
]


def grid_specs() -> pd.DataFrame:
    rows = [
        ("round2_industry_sue_w20", "sue_refined", 0.20, np.nan),
        ("round2_industry_sue_w30", "sue_refined", 0.30, np.nan),
        ("round2_industry_sue_w40", "sue_refined", 0.40, np.nan),
        ("round2_industry_gross_sue_w30", "sue_refined", 0.31, np.nan),
        ("round2_margin_yoy_industry", "margin", 0.00, np.nan),
        ("round2_gp_margin_w20", "margin", 0.20, np.nan),
        ("round2_gp_margin_w30", "margin", 0.30, np.nan),
        ("round2_cost_exp_profit_industry", "expense", 0.00, np.nan),
        ("round2_gp_cost_exp_profit_w30", "expense", 0.30, np.nan),
        ("round2_gp_low_period_exp_w30", "expense", 0.31, np.nan),
        ("round2_receivable_turnover_industry", "receivable", 0.00, np.nan),
        ("round2_gp_receivable_w20", "receivable", 0.20, np.nan),
        ("round2_gp_receivable_w30", "receivable", 0.30, np.nan),
        ("round2_gp_ar_turnover_w30", "receivable", 0.31, np.nan),
        ("round2_growth_quality_w20", "growth_quality", 0.20, np.nan),
        ("round2_growth_quality_w30", "growth_quality", 0.30, np.nan),
        ("round2_industry_growth_quality_w30", "growth_quality", 0.31, np.nan),
        ("round2_industry_sue_interaction", "interaction", 0.10, np.nan),
    ]
    return pd.DataFrame(rows, columns=["factor", "version", "q1_weight", "half_life"])


def load_events(dataset: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID", "ID", "PUBLISH_DATE", "END_DATE_REP", "END_DATE",
        "UPDATE_TIME", "REPORT_TYPE", *FIELDS,
    ]
    events = pd.read_parquet(dataset, columns=columns)
    for column in ["PUBLISH_DATE", "END_DATE_REP", "END_DATE", "UPDATE_TIME"]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    events = events.loc[
        events["END_DATE"].eq(events["END_DATE_REP"])
        & events["REPORT_TYPE"].isin(["Q1", "S1", "Q3", "A"])
    ].copy()
    quarter_map = {"Q1": 1, "S1": 2, "Q3": 3, "A": 4}
    events["FISCAL_QUARTER"] = events["REPORT_TYPE"].map(quarter_map)
    events["QUARTER_INDEX"] = events["END_DATE"].dt.year * 4 + events["FISCAL_QUARTER"]
    events["EVENT_TIME"] = events["PUBLISH_DATE"].dt.normalize() + pd.to_timedelta(86_399, unit="s")
    events = events.dropna(
        subset=["SECURITY_ID", "ID", "EVENT_TIME", "FISCAL_QUARTER", "QUARTER_INDEX"]
    ).copy()
    events["SECURITY_ID"] = pd.to_numeric(events["SECURITY_ID"]).astype("int64")
    events["ID"] = pd.to_numeric(events["ID"]).astype("int64")
    for column in FIELDS:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events = events.sort_values(
        ["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX", "UPDATE_TIME", "ID"]
    ).drop_duplicates(["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"], keep="last")
    return events


def prepare_events(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX", "ID"]
    )
    newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
    clean = available.loc[available["QUARTER_INDEX"].eq(newest)].copy()
    clean = clean.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
    return clean[
        ["SECURITY_ID", "AVAILABLE_DATE", "FISCAL_QUARTER", "QUARTER_INDEX", *FIELDS]
    ].reset_index(drop=True)


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    dense = _daily_median_fill(inputs, FIELDS + DENSE_FIELDS)
    z = robust_daily_zscore(dense, FIELDS + DENSE_FIELDS)
    gp_ind = _industry_robust_zscore(
        dense.assign(GP=z["GROSS_PROFIT_YOY"]), "GP", "INDUSTRY"
    )
    margin_ind = _industry_robust_zscore(
        dense.assign(MARGIN=z["NP_MARGIN_YOY"]), "MARGIN", "INDUSTRY"
    )
    cost_profit_ind = _industry_robust_zscore(
        dense.assign(COST_PROFIT=z["P_COST_EXP"]), "COST_PROFIT", "INDUSTRY"
    )
    receivable_ind = _industry_robust_zscore(
        dense.assign(AR_REC=z["AR_REC_R"]), "AR_REC", "INDUSTRY"
    )
    sue = z["dense_q1_earnings_sue_ensemble"]
    gross_sue = z["dense_q1_gross_profit_sue"]
    margin = z["NP_MARGIN_YOY"]
    cost_profit = z["P_COST_EXP"]
    low_period_expense = -z["PERIOD_EXP_TR"]
    receivable = z["AR_REC_R"]
    ar_turnover = z["AR_R"]
    cfo_growth = z["N_CF_OPA_YOY"]

    result = dense[KEYS].copy()
    for weight in (0.20, 0.30, 0.40):
        result[f"round2_industry_sue_w{int(weight*100)}"] = (1-weight) * gp_ind + weight * sue
    result["round2_industry_gross_sue_w30"] = 0.70 * gp_ind + 0.30 * gross_sue
    result["round2_margin_yoy_industry"] = margin_ind
    result["round2_gp_margin_w20"] = 0.80 * gp_ind + 0.20 * margin
    result["round2_gp_margin_w30"] = 0.70 * gp_ind + 0.30 * margin
    result["round2_cost_exp_profit_industry"] = cost_profit_ind
    result["round2_gp_cost_exp_profit_w30"] = 0.70 * gp_ind + 0.30 * cost_profit
    result["round2_gp_low_period_exp_w30"] = 0.70 * gp_ind + 0.30 * low_period_expense
    result["round2_receivable_turnover_industry"] = receivable_ind
    result["round2_gp_receivable_w20"] = 0.80 * gp_ind + 0.20 * receivable
    result["round2_gp_receivable_w30"] = 0.70 * gp_ind + 0.30 * receivable
    result["round2_gp_ar_turnover_w30"] = 0.70 * gp_ind + 0.30 * ar_turnover
    result["round2_growth_quality_w20"] = 0.80 * gp_ind + 0.10 * margin + 0.10 * cfo_growth
    result["round2_growth_quality_w30"] = 0.70 * gp_ind + 0.15 * margin + 0.15 * cfo_growth
    quality_raw = 0.70 * gp_ind + 0.15 * margin_ind + 0.15 * cfo_growth
    result["round2_industry_growth_quality_w30"] = quality_raw
    interaction = robust_daily_zscore(
        pd.DataFrame({"TRADE_DATE": dense["TRADE_DATE"], "X": gp_ind * sue}), ["X"]
    )["X"]
    result["round2_industry_sue_interaction"] = 0.65 * gp_ind + 0.25 * sue + 0.10 * interaction
    for factor in grid_specs()["factor"]:
        result[factor] = pd.to_numeric(result[factor], errors="coerce").astype("float32")
    return result


def generate_grid(
    panel_path: Path, indicator_path: Path, dense_path: Path, barra_path: Path,
    output_path: Path, manifest_path: Path, quality_path: Path,
) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    ordinal = pd.Series(np.arange(len(calendar), dtype=np.int16), index=calendar)
    events = prepare_events(load_events(indicator_path), calendar)
    specs = grid_specs()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    specs.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    reports = []
    try:
        for year in sorted(set(calendar.year)):
            filters = [
                ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
            ]
            panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=filters))
            mapped = map_events_to_panel(panel, events, ordinal)
            dense_factors = pd.read_parquet(
                dense_path, columns=KEYS + DENSE_FIELDS, filters=filters
            )
            barra = pd.read_parquet(barra_path, columns=KEYS + INDUSTRY_FACTORS, filters=filters)
            barra["TRADE_DATE"] = pd.to_datetime(barra["TRADE_DATE"]).dt.normalize()
            barra["INDUSTRY"] = _industry_labels(barra)
            inputs = mapped[KEYS + FIELDS].merge(
                dense_factors, on=KEYS, how="left", validate="one_to_one"
            ).merge(barra[KEYS + ["INDUSTRY"]], on=KEYS, how="left", validate="one_to_one")
            candidates = build_candidates(inputs)
            for factor in specs["factor"]:
                grouped = candidates.groupby("TRADE_DATE")[factor]
                missing = int(candidates[factor].isna().sum())
                constant = int(grouped.nunique(dropna=True).lt(2).sum())
                reports.append({"year": year, "factor": factor, "missing_rows": missing, "constant_days": constant})
                if missing or constant:
                    raise RuntimeError(f"{year} {factor}: missing={missing}, constant={constant}")
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            print(f"{year}: rows={len(candidates):,}, factors={len(specs)}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成候选")
    temporary.replace(output_path)
    pd.DataFrame(reports).to_csv(quality_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    root = BASE_DIR / "新测试结果" / "第二轮优化"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-grid")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument(
        "--indicator-pit", type=Path,
        default=BASE_DIR / "data" / "quarterly_financial_indicators" / "quarterly_financial_indicator_pit",
    )
    generate.add_argument("--dense", type=Path, default=BASE_DIR / "factor_components" / "dense_no_rank_factors.parquet")
    generate.add_argument("--barra", type=Path, default=BASE_DIR / "barra_diy.parquet")
    generate.add_argument("--output", type=Path, default=root / "round2_grid.parquet")
    generate.add_argument("--manifest", type=Path, default=root / "round2_grid.csv")
    generate.add_argument("--quality", type=Path, default=root / "round2_grid_quality.csv")
    fitted = sub.add_parser("fit")
    fitted.add_argument("--daily-ic", type=Path, required=True)
    fitted.add_argument("--manifest", type=Path, default=root / "round2_grid.csv")
    fitted.add_argument("--grid", type=Path, default=root / "round2_grid.parquet")
    fitted.add_argument("--output", type=Path, default=root / "round2_selected.parquet")
    fitted.add_argument("--report", type=Path, default=root / "round2_parameter_report.csv")
    fitted.add_argument("--parameters", type=Path, default=root / "round2_parameters.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-grid":
        generate_grid(
            args.panel.resolve(), args.indicator_pit.resolve(), args.dense.resolve(),
            args.barra.resolve(), args.output.resolve(), args.manifest.resolve(), args.quality.resolve(),
        )
    else:
        fit(
            args.daily_ic.resolve(), args.manifest.resolve(), args.grid.resolve(),
            args.output.resolve(), args.report.resolve(), args.parameters.resolve(),
        )


if __name__ == "__main__":
    main()

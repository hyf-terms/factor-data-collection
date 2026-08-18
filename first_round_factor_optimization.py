"""First-round dense optimization of the strongest strict financial factors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from coverage_literature_factor_search import (
    load_events,
    map_events_to_panel,
    prepare_available_events,
)
from dense_q1_gross_profit_factors import robust_daily_zscore, select_parameters
from event_financial_factor_search import KEYS, _normalize_panel
from factors_neus_only2 import INDUSTRY_FACTORS


BASE_DIR = Path(__file__).resolve().parent
EVENT_FIELDS = [
    "GROSS_PROFIT_YOY",
    "N_CF_OPA_NIA",
    "NI_ATTR_P_YOY",
    "N_CF_OPA_YOY",
]
FAMILIES = ["sue_enhanced", "cash_confirmed", "timing", "industry"]


def grid_specs() -> pd.DataFrame:
    rows = [
        ("round1_gp_sue_w70", "sue_enhanced", 0.30, np.nan),
        ("round1_gp_sue_w50", "sue_enhanced", 0.50, np.nan),
        ("round1_gp_cash_conversion_w70", "cash_confirmed", 0.30, np.nan),
        ("round1_gp_cash_conversion_w50", "cash_confirmed", 0.50, np.nan),
        ("round1_gp_cash_confirmation_w70", "cash_confirmed", 0.35, np.nan),
        ("round1_gp_time_floor50_h60", "timing", 0.50, 60.0),
        ("round1_gp_time_floor50_h120", "timing", 0.50, 120.0),
        ("round1_gp_time_floor75_h60", "timing", 0.25, 60.0),
        ("round1_gp_time_floor75_h120", "timing", 0.25, 120.0),
        ("round1_gp_industry_z", "industry", 0.00, np.nan),
        ("round1_gp_industry_sue_w70", "industry", 0.30, np.nan),
    ]
    return pd.DataFrame(rows, columns=["factor", "version", "q1_weight", "half_life"])


def _daily_median_fill(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        medians = values.groupby(result["TRADE_DATE"], sort=False).transform("median")
        if medians.loc[values.isna()].isna().any():
            raise ValueError(f"{column}存在整日全空，不能稠密化")
        result[column] = values.fillna(medians)
    return result


def _industry_labels(barra: pd.DataFrame) -> pd.Series:
    values = barra[INDUSTRY_FACTORS].fillna(0.0).to_numpy(dtype=np.float64)
    positions = np.argmax(values, axis=1)
    maximum = values[np.arange(len(values)), positions]
    labels = np.asarray(INDUSTRY_FACTORS, dtype=object)[positions]
    labels[maximum <= 0] = "未分类"
    return pd.Series(labels, index=barra.index, dtype="object")


def _industry_robust_zscore(
    frame: pd.DataFrame, value_column: str, industry_column: str
) -> pd.Series:
    keys = [frame["TRADE_DATE"], frame[industry_column]]
    values = pd.to_numeric(frame[value_column], errors="coerce")
    center = values.groupby(keys, sort=False).transform("median")
    deviation = (values - center).abs()
    mad = deviation.groupby(keys, sort=False).transform("median")
    scale = 1.4826 * mad
    z = (values - center) / scale.where(scale.gt(1e-12))
    fallback = robust_daily_zscore(
        pd.DataFrame({"TRADE_DATE": frame["TRADE_DATE"], value_column: values}),
        [value_column],
    )[value_column]
    return z.replace([np.inf, -np.inf], np.nan).fillna(fallback).fillna(0.0)


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    raw_columns = EVENT_FIELDS + ["dense_q1_earnings_sue_ensemble"]
    dense = _daily_median_fill(inputs, raw_columns)
    z = robust_daily_zscore(dense, raw_columns)
    gp = z["GROSS_PROFIT_YOY"]
    sue = z["dense_q1_earnings_sue_ensemble"]
    cash_conversion = z["N_CF_OPA_NIA"]
    cash_confirmation = pd.concat(
        [z["GROSS_PROFIT_YOY"], z["NI_ATTR_P_YOY"], z["N_CF_OPA_YOY"]],
        axis=1,
    ).mean(axis=1)
    industry_gp = _industry_robust_zscore(
        dense.assign(GP_Z=gp), "GP_Z", "INDUSTRY"
    )
    age = pd.to_numeric(dense["EVENT_AGE"], errors="coerce").clip(lower=0)

    result = dense[KEYS].copy()
    result["round1_gp_sue_w70"] = 0.70 * gp + 0.30 * sue
    result["round1_gp_sue_w50"] = 0.50 * gp + 0.50 * sue
    result["round1_gp_cash_conversion_w70"] = 0.70 * gp + 0.30 * cash_conversion
    result["round1_gp_cash_conversion_w50"] = 0.50 * gp + 0.50 * cash_conversion
    result["round1_gp_cash_confirmation_w70"] = 0.65 * gp + 0.35 * cash_confirmation
    for floor in (0.50, 0.75):
        for half_life in (60, 120):
            multiplier = floor + (1.0 - floor) * np.exp(
                -np.log(2.0) * age / half_life
            ).fillna(floor)
            result[f"round1_gp_time_floor{int(floor*100)}_h{half_life}"] = gp * multiplier
    result["round1_gp_industry_z"] = industry_gp
    result["round1_gp_industry_sue_w70"] = 0.70 * industry_gp + 0.30 * sue
    for factor in grid_specs()["factor"]:
        result[factor] = pd.to_numeric(result[factor], errors="coerce").astype("float32")
    return result


def generate_grid(
    panel_path: Path,
    indicator_path: Path,
    sue_path: Path,
    barra_path: Path,
    output_path: Path,
    manifest_path: Path,
    quality_path: Path,
) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    ordinal = pd.Series(np.arange(len(calendar), dtype=np.int16), index=calendar)
    events = prepare_available_events(load_events(indicator_path), calendar)
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
            sue = pd.read_parquet(
                sue_path,
                columns=KEYS + ["dense_q1_earnings_sue_ensemble"],
                filters=filters,
            )
            barra = pd.read_parquet(
                barra_path, columns=KEYS + INDUSTRY_FACTORS, filters=filters
            )
            barra["TRADE_DATE"] = pd.to_datetime(barra["TRADE_DATE"]).dt.normalize()
            barra["INDUSTRY"] = _industry_labels(barra)
            inputs = mapped[KEYS + ["EVENT_AGE", *EVENT_FIELDS]].merge(
                sue, on=KEYS, how="left", validate="one_to_one"
            ).merge(
                barra[KEYS + ["INDUSTRY"]], on=KEYS, how="left", validate="one_to_one"
            )
            candidates = build_candidates(inputs)
            for factor in specs["factor"]:
                grouped = candidates.groupby("TRADE_DATE")[factor]
                constant_days = int(grouped.nunique(dropna=True).lt(2).sum())
                missing = int(candidates[factor].isna().sum())
                reports.append(
                    {"year": year, "factor": factor, "missing_rows": missing, "constant_days": constant_days}
                )
                if missing or constant_days:
                    raise RuntimeError(
                        f"{year} {factor}不满足严格条件: missing={missing}, constant_days={constant_days}"
                    )
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
    os.replace(temporary, output_path)
    pd.DataFrame(reports).to_csv(quality_path, index=False, encoding="utf-8-sig")


def fit(
    daily_ic_path: Path,
    manifest_path: Path,
    grid_path: Path,
    output_path: Path,
    report_path: Path,
    parameters_path: Path,
) -> None:
    daily = pd.read_parquet(daily_ic_path)
    specs = pd.read_csv(manifest_path)
    selected, report = select_parameters(daily, specs)
    source = selected["factor"].tolist()
    names = [f"optimized_{version}" for version in selected["version"]]
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    parquet = pq.ParquetFile(grid_path)
    try:
        for batch in parquet.iter_batches(columns=KEYS + source, batch_size=250_000):
            table = pa.Table.from_batches([batch]).rename_columns(KEYS + names)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("参数网格为空")
    os.replace(temporary, output_path)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    parameters_path.write_text(
        json.dumps(selected.to_dict("records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(report.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-grid")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument(
        "--indicator-pit", type=Path,
        default=BASE_DIR / "data" / "quarterly_financial_indicators" / "quarterly_financial_indicator_pit",
    )
    generate.add_argument(
        "--sue", type=Path,
        default=BASE_DIR / "factor_components" / "dense_no_rank_factors.parquet",
    )
    generate.add_argument("--barra", type=Path, default=BASE_DIR / "barra_diy.parquet")
    generate.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_grid.parquet",
    )
    generate.add_argument(
        "--manifest", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_grid.csv",
    )
    generate.add_argument(
        "--quality", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_grid_quality.csv",
    )
    fitted = sub.add_parser("fit")
    fitted.add_argument("--daily-ic", type=Path, required=True)
    fitted.add_argument(
        "--manifest", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_grid.csv",
    )
    fitted.add_argument(
        "--grid", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_grid.parquet",
    )
    fitted.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_selected.parquet",
    )
    fitted.add_argument(
        "--report", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_parameter_report.csv",
    )
    fitted.add_argument(
        "--parameters", type=Path,
        default=BASE_DIR / "新测试结果" / "第一轮优化" / "round1_parameters.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-grid":
        generate_grid(
            args.panel.resolve(), args.indicator_pit.resolve(), args.sue.resolve(),
            args.barra.resolve(), args.output.resolve(), args.manifest.resolve(),
            args.quality.resolve(),
        )
    else:
        fit(
            args.daily_ic.resolve(), args.manifest.resolve(), args.grid.resolve(),
            args.output.resolve(), args.report.resolve(), args.parameters.resolve(),
        )


if __name__ == "__main__":
    main()

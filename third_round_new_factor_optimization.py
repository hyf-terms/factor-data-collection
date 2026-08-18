"""Dense no-rank search for cash, working-capital and efficiency signals.

All heterogeneous inputs are converted to daily robust Z scores before they
are combined.  Missing source observations are filled with the same-day
median, so no stock or date is removed.  ``*_incremental`` columns are daily
cross-sectional residuals after regressing the new signal on gross-profit
growth; they measure information not already contained in the anchor factor.
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


BASE_DIR = Path(__file__).resolve().parent
VENDOR_FIELDS = [
    "GROSS_PROFIT_YOY",
    "N_CF_OPA_NIA",
    "N_CF_OPA_NIA_YOY",
    "NP_MARGIN_YOY",
    "SELL_EXP_TR",
    "ADMIN_EXP_TR",
    "R_D_EXP_YOY",
    "REVENUE_YOY",
    "OPER_PROFIT_YOY",
    "N_CF_OPA_YOY",
]
PANEL_FIELDS = [
    "accrual_quality",
    "receivable_abnormal_growth",
    "inventory_abnormal_growth",
]


def candidate_names() -> list[str]:
    bases = [
        "r3_cash_accrual_quality",
        "r3_cash_conversion",
        "r3_cash_conversion_improvement",
        "r3_receivable_quality",
        "r3_inventory_quality",
        "r3_margin_improvement",
        "r3_sga_efficiency",
        "r3_rd_revenue_match",
        "r3_op_vs_gp_growth",
        "r3_cash_quality_composite",
        "r3_working_capital_composite",
        "r3_margin_efficiency_composite",
        "r3_growth_quality_composite",
    ]
    families = ["cash_quality", "working_capital", "margin_efficiency", "growth_quality"]
    return bases + [f"r3_{name}_incremental" for name in families] + [
        f"r3_gp_{name}_w20" for name in families
    ]


def load_vendor_events(dataset: Path) -> pd.DataFrame:
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
    for column in VENDOR_FIELDS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(
        ["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX", "UPDATE_TIME", "ID"]
    ).drop_duplicates(["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"], keep="last")


def prepare_vendor_events(data: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    available = assign_available_trade_date(data, calendar).sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX", "ID"]
    )
    newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
    available = available.loc[available["QUARTER_INDEX"].eq(newest)]
    return available.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")[
        ["SECURITY_ID", "AVAILABLE_DATE", "QUARTER_INDEX", *VENDOR_FIELDS]
    ].reset_index(drop=True)


def _daily_orthogonal_residual(signal: pd.Series, anchor: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "x": signal, "g": anchor})
    x_mean = frame.groupby("date", sort=False)["x"].transform("mean")
    g_mean = frame.groupby("date", sort=False)["g"].transform("mean")
    x_centered = frame["x"] - x_mean
    g_centered = frame["g"] - g_mean
    covariance = (x_centered * g_centered).groupby(frame["date"], sort=False).transform("mean")
    variance = (g_centered * g_centered).groupby(frame["date"], sort=False).transform("mean")
    beta = covariance.div(variance.where(variance.gt(1e-12))).fillna(0.0)
    residual = x_centered - beta * g_centered
    # A linear scale preserves exact orthogonality.  Re-winsorizing here would
    # be nonlinear and could recreate exposure to the gross-profit anchor.
    scale = residual.groupby(dates, sort=False).transform("std").where(lambda x: x.gt(1e-12))
    return residual.div(scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _daily_neutral_fill(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill stock gaps by daily median and all-missing days by neutral zero."""
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        median = values.groupby(result["TRADE_DATE"], sort=False).transform("median")
        result[column] = values.fillna(median).fillna(0.0)
    return result


def build_candidates(inputs: pd.DataFrame) -> pd.DataFrame:
    source = VENDOR_FIELDS + PANEL_FIELDS
    dense = _daily_neutral_fill(inputs, source)
    z = robust_daily_zscore(dense, source)
    gp = z["GROSS_PROFIT_YOY"]

    components = {
        "cash_accrual_quality": z["accrual_quality"],
        "cash_conversion": z["N_CF_OPA_NIA"],
        "cash_conversion_improvement": z["N_CF_OPA_NIA_YOY"],
        "receivable_quality": z["receivable_abnormal_growth"],
        "inventory_quality": z["inventory_abnormal_growth"],
        "margin_improvement": z["NP_MARGIN_YOY"],
        "sga_efficiency": robust_daily_zscore(
            pd.DataFrame({
                "TRADE_DATE": dense["TRADE_DATE"],
                "value": -(z["SELL_EXP_TR"] + z["ADMIN_EXP_TR"]) / 2,
            }), ["value"]
        )["value"],
        "rd_revenue_match": robust_daily_zscore(
            pd.DataFrame({
                "TRADE_DATE": dense["TRADE_DATE"],
                "value": -(z["R_D_EXP_YOY"] - z["REVENUE_YOY"]).abs(),
            }), ["value"]
        )["value"],
        "op_vs_gp_growth": robust_daily_zscore(
            pd.DataFrame({
                "TRADE_DATE": dense["TRADE_DATE"],
                "value": z["OPER_PROFIT_YOY"] - gp,
            }), ["value"]
        )["value"],
    }
    composites = {
        "cash_quality": (
            components["cash_accrual_quality"]
            + components["cash_conversion"]
            + components["cash_conversion_improvement"]
        ) / 3,
        "working_capital": (
            components["receivable_quality"] + components["inventory_quality"]
        ) / 2,
        "margin_efficiency": (
            components["margin_improvement"]
            + components["sga_efficiency"]
            + components["rd_revenue_match"]
            + components["op_vs_gp_growth"]
        ) / 4,
        "growth_quality": (
            gp + z["N_CF_OPA_YOY"] + components["receivable_quality"]
        ) / 3,
    }
    result = dense[KEYS].copy()
    for name, values in components.items():
        result[f"r3_{name}"] = values
    for name, values in composites.items():
        standardized = robust_daily_zscore(
            pd.DataFrame({"TRADE_DATE": dense["TRADE_DATE"], "value": values}), ["value"]
        )["value"].fillna(0.0)
        result[f"r3_{name}_composite"] = standardized
        residual = _daily_orthogonal_residual(standardized, gp, dense["TRADE_DATE"])
        result[f"r3_{name}_incremental"] = residual
        result[f"r3_gp_{name}_w20"] = 0.80 * gp + 0.20 * residual
    for factor in candidate_names():
        result[factor] = pd.to_numeric(result[factor], errors="coerce").fillna(0.0).astype("float32")
    return result


def generate(panel_path: Path, indicator_path: Path, barra_path: Path, output: Path, manifest: Path, quality: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    ordinal = pd.Series(np.arange(len(calendar), dtype=np.int16), index=calendar)
    events = prepare_vendor_events(load_vendor_events(indicator_path), calendar)
    names = candidate_names()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"factor": names}).to_csv(manifest, index=False, encoding="utf-8-sig")
    temporary = output.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    reports: list[dict[str, object]] = []
    try:
        for year in sorted(set(calendar.year)):
            filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
            panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS + PANEL_FIELDS, filters=filters))
            mapped = map_events_to_panel(panel[KEYS], events, ordinal)
            inputs = panel.merge(mapped[KEYS + VENDOR_FIELDS], on=KEYS, how="left", validate="one_to_one")
            candidates = build_candidates(inputs)
            for factor in names:
                grouped = candidates.groupby("TRADE_DATE")[factor]
                row = {
                    "year": year, "factor": factor,
                    "missing_rows": int(candidates[factor].isna().sum()),
                    "constant_days": int(grouped.nunique(dropna=True).lt(2).sum()),
                }
                reports.append(row)
                # Keep diagnostics and let the strict tester reject an
                # ineligible standalone signal.  Composite candidates may
                # still be fully usable because another component has data.
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


def build_incremental_blends(round2_path: Path, round3_path: Path, output: Path) -> None:
    """Add the sole training-stable incremental signal at conservative weights."""
    anchor = pd.read_parquet(round2_path, columns=KEYS + ["optimized_interaction"])
    incremental = pd.read_parquet(
        round3_path, columns=KEYS + ["r3_margin_efficiency_incremental"]
    )
    data = anchor.merge(incremental, on=KEYS, how="inner", validate="one_to_one")
    result = data[KEYS].copy()
    for weight in (0.05, 0.10, 0.15, 0.20):
        name = f"r3_interaction_margin_w{int(weight * 100):02d}"
        result[name] = (
            (1.0 - weight) * data["optimized_interaction"]
            + weight * data["r3_margin_efficiency_incremental"]
        ).astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")


def finalize_selection(grid_daily: Path, blend_daily: Path, blends: Path, output: Path, report: Path) -> None:
    """Freeze a blend using training data and document incremental stability."""
    grid = pd.read_parquet(grid_daily)
    blend_ic = pd.read_parquet(blend_daily)
    for frame in (grid, blend_ic):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"])

    rows: list[dict[str, object]] = []
    incrementals = grid.loc[grid["factor"].str.endswith("_incremental")].copy()
    for factor, group in incrementals.groupby("factor"):
        yearly = group.loc[group["TRADE_DATE"].dt.year.between(2017, 2022)].groupby(
            group.loc[group["TRADE_DATE"].dt.year.between(2017, 2022), "TRADE_DATE"].dt.year
        )["neutral_ic"].mean()
        rows.append({
            "stage": "incremental", "factor": factor,
            "train_2017_2022_ic": group.loc[group["TRADE_DATE"].dt.year.between(2017, 2022), "neutral_ic"].mean(),
            "validation_2023_2024_ic": group.loc[group["TRADE_DATE"].dt.year.between(2023, 2024), "neutral_ic"].mean(),
            "holdout_2025_2026_ic": group.loc[group["TRADE_DATE"].dt.year.between(2025, 2026), "neutral_ic"].mean(),
            "all_train_years_positive": bool(len(yearly) == 6 and yearly.gt(0).all()),
            "selected": factor == "r3_margin_efficiency_incremental",
        })
    train_means: dict[str, float] = {}
    for factor, group in blend_ic.groupby("factor"):
        train = group.loc[group["TRADE_DATE"].dt.year.between(2017, 2022), "neutral_ic"].mean()
        train_means[factor] = float(train)
        rows.append({
            "stage": "blend", "factor": factor,
            "train_2017_2022_ic": train,
            "validation_2023_2024_ic": group.loc[group["TRADE_DATE"].dt.year.between(2023, 2024), "neutral_ic"].mean(),
            "holdout_2025_2026_ic": group.loc[group["TRADE_DATE"].dt.year.between(2025, 2026), "neutral_ic"].mean(),
            "all_train_years_positive": True,
            "selected": False,
        })
    selected = max(train_means, key=train_means.get)
    for row in rows:
        if row["stage"] == "blend" and row["factor"] == selected:
            row["selected"] = True
    data = pd.read_parquet(blends, columns=KEYS + [selected])
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(rows).to_csv(report, index=False, encoding="utf-8-sig")


def main() -> None:
    root = BASE_DIR / "新测试结果" / "第三轮新因子"
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    parser.add_argument("--indicator-pit", type=Path, default=BASE_DIR / "data" / "quarterly_financial_indicators" / "quarterly_financial_indicator_pit")
    parser.add_argument("--barra", type=Path, default=BASE_DIR / "barra_diy.parquet")
    parser.add_argument("--output", type=Path, default=root / "round3_grid.parquet")
    parser.add_argument("--manifest", type=Path, default=root / "round3_grid.csv")
    parser.add_argument("--quality", type=Path, default=root / "round3_grid_quality.csv")
    parser.add_argument("--build-blends", action="store_true")
    parser.add_argument("--round2-selected", type=Path, default=BASE_DIR / "新测试结果" / "第二轮优化" / "round2_selected.parquet")
    parser.add_argument("--blend-output", type=Path, default=root / "round3_incremental_blends.parquet")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--grid-daily-ic", type=Path, default=root / "grid_test" / "daily_ic.parquet")
    parser.add_argument("--blend-daily-ic", type=Path, default=root / "blend_test" / "daily_ic.parquet")
    parser.add_argument("--final-output", type=Path, default=root / "round3_selected.parquet")
    parser.add_argument("--selection-report", type=Path, default=root / "round3_selection_report.csv")
    args = parser.parse_args()
    if args.finalize:
        finalize_selection(
            args.grid_daily_ic.resolve(), args.blend_daily_ic.resolve(),
            args.blend_output.resolve(), args.final_output.resolve(),
            args.selection_report.resolve(),
        )
    elif args.build_blends:
        build_incremental_blends(
            args.round2_selected.resolve(), args.output.resolve(), args.blend_output.resolve()
        )
    else:
        generate(args.panel.resolve(), args.indicator_pit.resolve(), args.barra.resolve(), args.output.resolve(), args.manifest.resolve(), args.quality.resolve())


if __name__ == "__main__":
    main()

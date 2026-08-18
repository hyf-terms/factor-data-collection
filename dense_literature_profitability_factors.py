"""Build dense PIT profitability, cash-profitability, accrual, and investment factors.

The batch contains direct accounting-characteristic reproductions from
Novy-Marx (2013), Fama and French (2015), Ball et al. (2016), Sloan (1996),
and Hou, Xue, and Zhang (2015).  Annual versions follow the papers as closely
as the Chinese statements allow.  TTM versions are explicitly labelled local
PIT adaptations: the newest four disclosed standalone quarters remain active
until a newer quarter becomes available.

No cross-sectional percentile rank is used.  Missing stocks are assigned the
same-date median only after the raw PIT signal has been mapped to the complete
label universe.  This keeps every trading date and stock while preserving all
observed accounting values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import (
    COMMON_COLUMNS,
    FINANCIAL_INDUSTRIES,
    REPORT_QUARTERS,
    REPORT_TYPES,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
INCOME_FIELDS = [
    "N_INCOME_ATTR_P",
    "REVENUE",
    "COGS",
    "SELL_EXP",
    "ADMIN_EXP",
    "INT_EXP_FINAN_EXP",
]
CASHFLOW_FIELDS = ["N_CF_OPERATE_A"]
BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY",
    "T_ASSETS",
    "T_EQUITY_ATTR_P",
    "AR",
    "INVENTORIES",
    "PREPAYMENT",
    "DEFER_REVENUE",
    "AP",
    "ACCRUED_EXP",
]
ZERO_IF_UNDISCLOSED = {
    "SELL_EXP",
    "ADMIN_EXP",
    "INT_EXP_FINAN_EXP",
    "AR",
    "INVENTORIES",
    "PREPAYMENT",
    "DEFER_REVENUE",
    "AP",
    "ACCRUED_EXP",
}

ANNUAL_COLUMNS = [
    "dense_lit_nm_gp_assets_annual",
    "dense_lit_ball_op_assets_annual",
    "dense_lit_ball_cbop_assets_annual",
    "dense_lit_ff_op_book_equity_annual",
    "dense_lit_sloan_low_accruals_annual",
    "dense_lit_ff_low_asset_growth_annual",
]
QUARTERLY_COLUMNS = [
    "dense_lit_hxz_qroe",
    "dense_lit_qroa",
    "dense_lit_qcfoa",
    "dense_lit_q_low_accruals",
    "dense_lit_q_gp_assets",
    "dense_lit_q_op_assets",
    "dense_lit_q_ff_op_book_equity",
    "dense_lit_q_low_asset_growth",
    "dense_lit_q_gross_margin",
    "dense_lit_ttm_gp_assets",
    "dense_lit_ttm_op_assets",
    "dense_lit_ttm_cbop_assets",
    "dense_lit_ttm_ff_op_book_equity",
    "dense_lit_ttm_cfoa",
    "dense_lit_ttm_low_accruals",
    "dense_lit_ttm_low_asset_growth",
]
CANDIDATE_COLUMNS = ANNUAL_COLUMNS + QUARTERLY_COLUMNS


def _latest_time(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in columns],
        axis=1,
    ).max(axis=1)


def _read_inputs(pit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=COMMON_COLUMNS + INCOME_FIELDS,
        engine="pyarrow",
    )
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + CASHFLOW_FIELDS,
        engine="pyarrow",
    )
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    for field in ZERO_IF_UNDISCLOSED.intersection(income.columns):
        income[field] = pd.to_numeric(income[field], errors="coerce").fillna(0.0)
    for field in ZERO_IF_UNDISCLOSED.intersection(balance.columns):
        balance[field] = pd.to_numeric(balance[field], errors="coerce").fillna(0.0)
    return income, cashflow, balance


def build_quarterly_balance_extended(balance: pd.DataFrame) -> pd.DataFrame:
    required = set(COMMON_COLUMNS + BALANCE_FIELDS)
    missing = sorted(required.difference(balance.columns))
    if missing:
        raise KeyError(f"资产负债表PIT缺少字段: {missing}")
    data = balance[COMMON_COLUMNS + BALANCE_FIELDS].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    for column in BALANCE_FIELDS:
        if column != "INDUSTRY_CATEGORY":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(
        subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS"]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    data["QUARTER_INDEX"] = data["FISCAL_YEAR"].astype("int64") * 4 + data["FISCAL_QUARTER"]
    key = ["SECURITY_ID", "QUARTER_INDEX"]
    data = data.sort_values(key + ["ACT_PUBTIME", "ID"]).drop_duplicates(key, keep="first")
    return data[
        key + ["FISCAL_YEAR", "FISCAL_QUARTER", "END_DATE", "ACT_PUBTIME", *BALANCE_FIELDS]
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def _standalone_flows(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for field in INCOME_FIELDS:
        result[field] = build_standalone_quarterly_metric(
            income, field, name="利润表PIT"
        )
    for field in CASHFLOW_FIELDS:
        result[field] = build_standalone_quarterly_metric(
            cashflow, field, name="现金流量表PIT"
        )
    return result


def _ttm(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    data = frame.sort_values(["SECURITY_ID", "QUARTER_INDEX"]).reset_index(drop=True).copy()
    discontinuity = data.groupby("SECURITY_ID")["QUARTER_INDEX"].diff().ne(1)
    data["_SEGMENT"] = discontinuity.groupby(data["SECURITY_ID"]).cumsum()
    groups = [data["SECURITY_ID"], data["_SEGMENT"]]
    data[f"TTM_{field}"] = (
        data.groupby(groups, sort=False)[field]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .reindex(data.index)
    )
    event_time = pd.to_datetime(data["EVENT_TIME"], errors="coerce")
    time_ns = pd.Series(
        event_time.astype("int64").to_numpy(dtype=np.float64), index=data.index
    ).where(event_time.notna())
    latest_ns = (
        time_ns.groupby(groups, sort=False)
        .rolling(4, min_periods=4)
        .max()
        .reset_index(level=[0, 1], drop=True)
        .reindex(data.index)
    )
    data[f"TTM_{field}_EVENT_TIME"] = pd.to_datetime(latest_ns, unit="ns", errors="coerce")
    return data[
        ["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX", f"TTM_{field}", f"TTM_{field}_EVENT_TIME"]
    ].dropna(subset=[f"TTM_{field}", f"TTM_{field}_EVENT_TIME"])


def _merge_flow_tables(
    flow_tables: dict[str, pd.DataFrame],
    fields: list[str],
    *,
    ttm: bool,
) -> pd.DataFrame:
    keys = ["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX"]
    tables = {field: _ttm(flow_tables[field], field) if ttm else flow_tables[field] for field in fields}
    first_field = fields[0]
    value_name = f"TTM_{first_field}" if ttm else first_field
    time_name = f"TTM_{first_field}_EVENT_TIME" if ttm else "EVENT_TIME"
    base = tables[first_field][keys + [value_name, time_name]].rename(
        columns={time_name: f"{first_field}_EVENT_TIME"}
    )
    for field in fields[1:]:
        value_name = f"TTM_{field}" if ttm else field
        time_name = f"TTM_{field}_EVENT_TIME" if ttm else "EVENT_TIME"
        other = tables[field][keys + [value_name, time_name]].rename(
            columns={time_name: f"{field}_EVENT_TIME"}
        )
        base = base.merge(other, on=keys, how="inner", validate="one_to_one")
    base["FLOW_EVENT_TIME"] = _latest_time(base, [f"{field}_EVENT_TIME" for field in fields])
    return base


def _lag_balance(balance: pd.DataFrame, quarters: int, prefix: str) -> pd.DataFrame:
    values = [field for field in BALANCE_FIELDS if field != "INDUSTRY_CATEGORY"]
    lagged = balance[["SECURITY_ID", "QUARTER_INDEX", "BALANCE_EVENT_TIME", *values]].copy()
    lagged["QUARTER_INDEX"] += quarters
    return lagged.rename(
        columns={
            "BALANCE_EVENT_TIME": f"{prefix}_BALANCE_EVENT_TIME",
            **{field: f"{prefix}_{field}" for field in values},
        }
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.gt(0) & np.isfinite(denominator)
    return numerator.div(denominator).where(valid).replace([np.inf, -np.inf], np.nan)


def _long_event(
    frame: pd.DataFrame,
    factor: str,
    value: pd.Series,
    event_columns: list[str],
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    result["EVENT_TIME"] = _latest_time(frame, event_columns)
    result["factor"] = factor
    result["value"] = pd.to_numeric(value, errors="coerce")
    return result.dropna(subset=["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", "value"])


def calculate_factor_events(
    flow_tables: dict[str, pd.DataFrame],
    balance: pd.DataFrame,
) -> pd.DataFrame:
    """Return long-form raw factor events before daily neutral filling."""
    q_fields = ["N_INCOME_ATTR_P", "N_CF_OPERATE_A"]
    q = _merge_flow_tables(flow_tables, q_fields, ttm=False)
    q = q.merge(balance, on=["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX"], how="inner")
    q = q.merge(_lag_balance(balance, 1, "L1"), on=["SECURITY_ID", "QUARTER_INDEX"], how="inner")
    q_event_columns = ["FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L1_BALANCE_EVENT_TIME"]
    q_den_assets = q["L1_T_ASSETS"]
    events = [
        _long_event(q, "dense_lit_hxz_qroe", _safe_ratio(q["N_INCOME_ATTR_P"], q["L1_T_EQUITY_ATTR_P"]), q_event_columns),
        _long_event(q, "dense_lit_qroa", _safe_ratio(q["N_INCOME_ATTR_P"], q_den_assets), q_event_columns),
        _long_event(q, "dense_lit_qcfoa", _safe_ratio(q["N_CF_OPERATE_A"], q_den_assets), q_event_columns),
        _long_event(q, "dense_lit_q_low_accruals", _safe_ratio(q["N_CF_OPERATE_A"] - q["N_INCOME_ATTR_P"], q_den_assets), q_event_columns),
    ]

    q_profit_fields = ["REVENUE", "COGS", "SELL_EXP", "ADMIN_EXP", "INT_EXP_FINAN_EXP"]
    q_profit = _merge_flow_tables(flow_tables, q_profit_fields, ttm=False)
    q_profit = q_profit.merge(
        balance,
        on=["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX"],
        how="inner",
    )
    q_profit = q_profit.merge(
        _lag_balance(balance, 1, "L1"),
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
    )
    q_profit_events = ["FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L1_BALANCE_EVENT_TIME"]
    q_gp = q_profit["REVENUE"] - q_profit["COGS"]
    q_op = q_gp - q_profit["SELL_EXP"] - q_profit["ADMIN_EXP"]
    q_ff_op = q_op - q_profit["INT_EXP_FINAN_EXP"]
    events.extend(
        [
            _long_event(q_profit, "dense_lit_q_gp_assets", _safe_ratio(q_gp, q_profit["L1_T_ASSETS"]), q_profit_events),
            _long_event(q_profit, "dense_lit_q_op_assets", _safe_ratio(q_op, q_profit["L1_T_ASSETS"]), q_profit_events),
            _long_event(q_profit, "dense_lit_q_ff_op_book_equity", _safe_ratio(q_ff_op, q_profit["L1_T_EQUITY_ATTR_P"]), q_profit_events),
            _long_event(q_profit, "dense_lit_q_low_asset_growth", -(q_profit["T_ASSETS"] / q_profit["L1_T_ASSETS"] - 1.0), q_profit_events),
            _long_event(q_profit, "dense_lit_q_gross_margin", _safe_ratio(q_gp, q_profit["REVENUE"]), q_profit_events),
        ]
    )

    ttm_fields = INCOME_FIELDS + CASHFLOW_FIELDS
    ttm = _merge_flow_tables(flow_tables, ttm_fields, ttm=True)
    ttm = ttm.merge(balance, on=["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX"], how="inner")
    ttm = ttm.merge(_lag_balance(balance, 4, "L4"), on=["SECURITY_ID", "QUARTER_INDEX"], how="inner")
    ttm_events = ["FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]
    gross_profit = ttm["TTM_REVENUE"] - ttm["TTM_COGS"]
    operating_profit = gross_profit - ttm["TTM_SELL_EXP"] - ttm["TTM_ADMIN_EXP"]
    ff_operating_profit = operating_profit - ttm["TTM_INT_EXP_FINAN_EXP"]
    delta_ar = ttm["AR"] - ttm["L4_AR"]
    delta_inventory = ttm["INVENTORIES"] - ttm["L4_INVENTORIES"]
    delta_prepay = ttm["PREPAYMENT"] - ttm["L4_PREPAYMENT"]
    delta_deferred = ttm["DEFER_REVENUE"] - ttm["L4_DEFER_REVENUE"]
    delta_ap = ttm["AP"] - ttm["L4_AP"]
    delta_accrued = ttm["ACCRUED_EXP"] - ttm["L4_ACCRUED_EXP"]
    cash_operating_profit = (
        operating_profit - delta_ar - delta_inventory - delta_prepay
        + delta_deferred + delta_ap + delta_accrued
    )
    ttm_values = {
        "dense_lit_ttm_gp_assets": _safe_ratio(gross_profit, ttm["T_ASSETS"]),
        "dense_lit_ttm_op_assets": _safe_ratio(operating_profit, ttm["L4_T_ASSETS"]),
        "dense_lit_ttm_cbop_assets": _safe_ratio(cash_operating_profit, ttm["L4_T_ASSETS"]),
        "dense_lit_ttm_ff_op_book_equity": _safe_ratio(ff_operating_profit, ttm["T_EQUITY_ATTR_P"]),
        "dense_lit_ttm_cfoa": _safe_ratio(ttm["TTM_N_CF_OPERATE_A"], ttm["L4_T_ASSETS"]),
        "dense_lit_ttm_low_accruals": _safe_ratio(ttm["TTM_N_CF_OPERATE_A"] - ttm["TTM_N_INCOME_ATTR_P"], ttm["L4_T_ASSETS"]),
        "dense_lit_ttm_low_asset_growth": -(ttm["T_ASSETS"] / ttm["L4_T_ASSETS"] - 1.0),
    }
    events.extend(_long_event(ttm, name, value, ttm_events) for name, value in ttm_values.items())

    annual = ttm.loc[ttm["FISCAL_QUARTER"].eq(4)].copy()
    annual_gp = annual["TTM_REVENUE"] - annual["TTM_COGS"]
    annual_op = annual_gp - annual["TTM_SELL_EXP"] - annual["TTM_ADMIN_EXP"]
    annual_ff_op = annual_op - annual["TTM_INT_EXP_FINAN_EXP"]
    annual_cbop = (
        annual_op
        - (annual["AR"] - annual["L4_AR"])
        - (annual["INVENTORIES"] - annual["L4_INVENTORIES"])
        - (annual["PREPAYMENT"] - annual["L4_PREPAYMENT"])
        + (annual["DEFER_REVENUE"] - annual["L4_DEFER_REVENUE"])
        + (annual["AP"] - annual["L4_AP"])
        + (annual["ACCRUED_EXP"] - annual["L4_ACCRUED_EXP"])
    )
    annual_values = {
        "dense_lit_nm_gp_assets_annual": _safe_ratio(annual_gp, annual["T_ASSETS"]),
        "dense_lit_ball_op_assets_annual": _safe_ratio(annual_op, annual["L4_T_ASSETS"]),
        "dense_lit_ball_cbop_assets_annual": _safe_ratio(annual_cbop, annual["L4_T_ASSETS"]),
        "dense_lit_ff_op_book_equity_annual": _safe_ratio(annual_ff_op, annual["T_EQUITY_ATTR_P"]),
        "dense_lit_sloan_low_accruals_annual": _safe_ratio(annual["TTM_N_CF_OPERATE_A"] - annual["TTM_N_INCOME_ATTR_P"], annual["L4_T_ASSETS"]),
        "dense_lit_ff_low_asset_growth_annual": -(annual["T_ASSETS"] / annual["L4_T_ASSETS"] - 1.0),
    }
    events.extend(_long_event(annual, name, value, ttm_events) for name, value in annual_values.items())
    result = pd.concat(events, ignore_index=True)
    result["QUARTER_INDEX"] = pd.to_numeric(result["QUARTER_INDEX"], errors="coerce").astype("int64")
    return result.sort_values(["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"]).reset_index(drop=True)


def prepare_wide_events(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for factor, group in events.groupby("factor", sort=False):
        available = assign_available_trade_date(group, calendar)
        available = available.sort_values(["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"])
        latest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        available = available.loc[available["QUARTER_INDEX"].eq(latest)]
        available = available.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        available["factor"] = factor
        pieces.append(available[["SECURITY_ID", "AVAILABLE_DATE", "factor", "value"]])
    long = pd.concat(pieces, ignore_index=True)
    wide = long.pivot_table(
        index=["SECURITY_ID", "AVAILABLE_DATE"], columns="factor", values="value", aggfunc="last"
    ).reset_index()
    for factor in CANDIDATE_COLUMNS:
        if factor not in wide:
            wide[factor] = np.nan
    wide = wide.sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[CANDIDATE_COLUMNS] = wide.groupby("SECURITY_ID", sort=False)[CANDIDATE_COLUMNS].ffill()
    return wide[["SECURITY_ID", "AVAILABLE_DATE", *CANDIDATE_COLUMNS]]


def map_and_fill(panel: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    mapped = pd.merge_asof(
        panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        events.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on="AVAILABLE_DATE",
        direction="backward",
    )
    missing = mapped[CANDIDATE_COLUMNS].isna().mean().to_dict()
    medians = mapped.groupby("TRADE_DATE")[CANDIDATE_COLUMNS].transform("median")
    mapped[CANDIDATE_COLUMNS] = mapped[CANDIDATE_COLUMNS].fillna(medians)
    if mapped[CANDIDATE_COLUMNS].isna().any().any():
        bad = mapped[CANDIDATE_COLUMNS].columns[mapped[CANDIDATE_COLUMNS].isna().any()].tolist()
        raise RuntimeError(f"中性填充后仍有空值: {bad}")
    for factor in CANDIDATE_COLUMNS:
        mapped[factor] = pd.to_numeric(mapped[factor], errors="coerce").astype("float32")
    return mapped[KEYS + CANDIDATE_COLUMNS].sort_values(KEYS), missing


def generate(
    pit_dir: Path,
    universe_path: Path,
    output_path: Path,
    event_path: Path,
    coverage_path: Path,
    metadata_path: Path,
) -> None:
    income, cashflow, balance_raw = _read_inputs(pit_dir)
    flows = _standalone_flows(income, cashflow)
    balance = build_quarterly_balance_extended(balance_raw)
    events = calculate_factor_events(flows, balance)
    dates = pd.read_parquet(universe_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    wide_events = prepare_wide_events(events, calendar)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    wide_events.to_parquet(event_path, index=False, compression="zstd")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    missing_weighted = {factor: 0.0 for factor in CANDIDATE_COLUMNS}
    total_rows = 0
    try:
        for year in sorted(set(calendar.year)):
            panel = pd.read_parquet(
                universe_path,
                columns=KEYS,
                filters=[("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))],
            )
            panel = _normalize_panel(panel)
            candidates, missing = map_and_fill(panel, wide_events)
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            for factor, rate in missing.items():
                missing_weighted[factor] += float(rate) * len(candidates)
            total_rows += len(candidates)
            print(f"{year}: rows={len(candidates):,}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成候选因子")
    os.replace(temporary, output_path)
    coverage = pd.DataFrame(
        {
            "factor": CANDIDATE_COLUMNS,
            "source_missing_rate_before_daily_median_fill": [missing_weighted[f] / total_rows for f in CANDIDATE_COLUMNS],
            "uses_rank": False,
            "daily_post_fill_non_null_rate": 1.0,
        }
    )
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    metadata = {
        "candidate_count": len(CANDIDATE_COLUMNS),
        "factor_event_rows": len(events),
        "wide_event_rows": len(wide_events),
        "panel_rows": total_rows,
        "uses_cross_sectional_rank": False,
        "missing_policy": "same-date median; observed PIT values unchanged",
        "availability": "latest disclosed annual or quarterly value persists until replacement; no fixed 60-day window",
        "financial_firms": "raw formulas excluded; same-date neutral median assigned in complete universe",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_eligible_subset(
    source: Path,
    coverage_path: Path,
    output: Path,
    *,
    maximum_source_missing: float = 0.10,
) -> list[str]:
    """Write only factors meeting the pre-fill source-coverage requirement."""
    coverage = pd.read_csv(coverage_path)
    selected = coverage.loc[
        coverage["source_missing_rate_before_daily_median_fill"].le(maximum_source_missing),
        "factor",
    ].tolist()
    if not selected:
        raise RuntimeError("没有候选满足源数据缺失率要求")
    parquet = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    temporary = output.with_suffix(".tmp.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch in parquet.iter_batches(columns=KEYS + selected, batch_size=250_000):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("源候选文件为空")
    os.replace(temporary, output)
    return selected


def parse_args() -> argparse.Namespace:
    root = BASE_DIR / "新测试结果" / "第七轮文献稠密财务因子"
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-existing", action="store_true")
    parser.add_argument("--pit-dir", type=Path, default=BASE_DIR / "data" / "new_pit")
    parser.add_argument("--universe", type=Path, default=BASE_DIR / "label.parquet")
    parser.add_argument("--output", type=Path, default=root / "dense_literature_candidates.parquet")
    parser.add_argument("--events", type=Path, default=root / "dense_literature_events.parquet")
    parser.add_argument("--coverage", type=Path, default=root / "dense_literature_coverage.csv")
    parser.add_argument("--metadata", type=Path, default=root / "dense_literature_metadata.json")
    parser.add_argument("--eligible-output", type=Path, default=root / "dense_literature_low_missing_candidates.parquet")
    parser.add_argument("--maximum-source-missing", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.filter_existing:
        selected = write_eligible_subset(
            args.output.resolve(),
            args.coverage.resolve(),
            args.eligible_output.resolve(),
            maximum_source_missing=args.maximum_source_missing,
        )
        print(f"eligible={len(selected)}: {', '.join(selected)}")
        return
    generate(
        args.pit_dir.resolve(),
        args.universe.resolve(),
        args.output.resolve(),
        args.events.resolve(),
        args.coverage.resolve(),
        args.metadata.resolve(),
    )
    selected = write_eligible_subset(
        args.output.resolve(),
        args.coverage.resolve(),
        args.eligible_output.resolve(),
        maximum_source_missing=args.maximum_source_missing,
    )
    print(f"eligible={len(selected)}: {', '.join(selected)}")


if __name__ == "__main__":
    main()

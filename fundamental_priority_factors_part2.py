"""Build the second batch of four PIT-safe financial factors.

Columns appended to ``factors.parquet``:

``asset_growth``
    Negative year-over-year total-asset growth.

``investment_to_assets``
    Negative change in fixed assets plus inventories, scaled by prior-year
    total assets.

``receivable_abnormal_growth``
    TTM revenue growth minus accounts-receivable growth.

``inventory_abnormal_growth``
    TTM cost-of-goods-sold growth minus inventory growth.

All four factors use the first valid PIT disclosure, become tradable no
earlier than the first market open after every required input was published,
exclude financial firms, and are winsorized at the daily 1st/99th percentiles.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fundamental_priority_factors import (
    WINSOR_LIMITS,
    _carry_one_factor,
    _latest_time,
    _normalize_panel,
    _winsorize_daily,
    build_ttm_metric,
)
from quarterly_f_score import (
    COMMON_COLUMNS,
    FINANCIAL_INDUSTRIES,
    REPORT_QUARTERS,
    REPORT_TYPES,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIT_DIR = BASE_DIR / "data" / "new_pit"
DEFAULT_FACTOR_PATH = BASE_DIR / "factors.parquet"
DEFAULT_AUDIT_DIR = BASE_DIR / "factor_components"

KEYS = ["TRADE_DATE", "SECURITY_ID"]
EVENT_KEYS = ["SECURITY_ID", "QUARTER_INDEX"]
FACTOR_COLUMNS = [
    "asset_growth",
    "investment_to_assets",
    "receivable_abnormal_growth",
    "inventory_abnormal_growth",
]
BALANCE_ITEMS = [
    "T_ASSETS",
    "FIXED_ASSETS_TOTAL",
    "INVENTORIES",
    "AR",
]
INCOME_ITEMS = ["REVENUE", "COGS"]


def _read_statements(pit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + ["INDUSTRY_CATEGORY"] + BALANCE_ITEMS,
        engine="pyarrow",
    )
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=COMMON_COLUMNS + INCOME_ITEMS,
        engine="pyarrow",
    )
    return balance, income


def build_balance_snapshot(
    balance: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """Return the earliest complete industrial-company quarterly snapshot."""
    required = set(
        COMMON_COLUMNS + ["INDUSTRY_CATEGORY"] + value_columns
    )
    missing = sorted(required.difference(balance.columns))
    if missing:
        raise KeyError(f"资产负债表PIT缺少字段: {missing}")

    columns = COMMON_COLUMNS + ["INDUSTRY_CATEGORY"] + value_columns
    data = balance[columns].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(
        data["SECURITY_ID"],
        errors="coerce",
    )
    for column in value_columns:
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
        subset=[
            "SECURITY_ID",
            "ACT_PUBTIME",
            "END_DATE",
        ]
        + value_columns
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = (
        data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    )
    data["QUARTER_INDEX"] = (
        data["FISCAL_YEAR"].astype("int64") * 4
        + data["FISCAL_QUARTER"]
    )
    data = data.sort_values(
        EVENT_KEYS + ["ACT_PUBTIME", "ID"]
    ).drop_duplicates(EVENT_KEYS, keep="first")
    if data.duplicated(EVENT_KEYS).any():
        raise ValueError("资产负债表季度快照存在重复键")
    return data[
        EVENT_KEYS
        + [
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "ACT_PUBTIME",
        ]
        + value_columns
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def _lag4_balance(
    snapshot: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    lagged = snapshot[
        EVENT_KEYS + ["BALANCE_EVENT_TIME"] + value_columns
    ].copy()
    lagged["QUARTER_INDEX"] += 4
    return lagged.rename(
        columns={
            "BALANCE_EVENT_TIME": "LAG4_BALANCE_EVENT_TIME",
            **{column: f"LAG4_{column}" for column in value_columns},
        }
    )


def build_asset_growth_events(balance: pd.DataFrame) -> pd.DataFrame:
    snapshot = build_balance_snapshot(balance, ["T_ASSETS"])
    lag4 = _lag4_balance(snapshot, ["T_ASSETS"])
    result = pd.merge(
        snapshot,
        lag4,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    result["asset_growth"] = (
        1.0 - result["T_ASSETS"].div(result["LAG4_T_ASSETS"])
    ).where(result["LAG4_T_ASSETS"].gt(0))
    result["EVENT_TIME"] = _latest_time(
        result,
        ["BALANCE_EVENT_TIME", "LAG4_BALANCE_EVENT_TIME"],
    )
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["asset_growth", "EVENT_TIME"]
    )
    return result[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "asset_growth",
        ]
    ].copy()


def build_investment_events(balance: pd.DataFrame) -> pd.DataFrame:
    items = ["T_ASSETS", "FIXED_ASSETS_TOTAL", "INVENTORIES"]
    snapshot = build_balance_snapshot(balance, items)
    lag4 = _lag4_balance(snapshot, items)
    result = pd.merge(
        snapshot,
        lag4,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    investment = (
        result["FIXED_ASSETS_TOTAL"]
        - result["LAG4_FIXED_ASSETS_TOTAL"]
        + result["INVENTORIES"]
        - result["LAG4_INVENTORIES"]
    )
    result["investment_to_assets"] = (
        -investment.div(result["LAG4_T_ASSETS"])
    ).where(result["LAG4_T_ASSETS"].gt(0))
    result["EVENT_TIME"] = _latest_time(
        result,
        ["BALANCE_EVENT_TIME", "LAG4_BALANCE_EVENT_TIME"],
    )
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["investment_to_assets", "EVENT_TIME"]
    )
    return result[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "investment_to_assets",
        ]
    ].copy()


def _build_ttm_income(
    income: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    quarterly = build_standalone_quarterly_metric(
        income,
        column,
        name="利润表PIT",
    )
    return build_ttm_metric(quarterly, column)


def _lag4_ttm(ttm: pd.DataFrame, column: str) -> pd.DataFrame:
    value = f"TTM_{column}"
    event = f"TTM_{column}_EVENT_TIME"
    lagged = ttm[EVENT_KEYS + [value, event]].copy()
    lagged["QUARTER_INDEX"] += 4
    return lagged.rename(
        columns={
            value: f"LAG4_{value}",
            event: f"LAG4_{event}",
        }
    )


def build_abnormal_growth_events(
    balance: pd.DataFrame,
    ttm_flow: pd.DataFrame,
    *,
    balance_column: str,
    flow_column: str,
    factor: str,
) -> pd.DataFrame:
    """Build flow-growth minus associated balance-item growth."""
    snapshot = build_balance_snapshot(
        balance,
        ["T_ASSETS", balance_column],
    )
    lag4_snapshot = _lag4_balance(
        snapshot,
        ["T_ASSETS", balance_column],
    )
    lag4_flow = _lag4_ttm(ttm_flow, flow_column)

    flow_value = f"TTM_{flow_column}"
    flow_event = f"TTM_{flow_column}_EVENT_TIME"
    base = pd.merge(
        snapshot,
        lag4_snapshot,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        ttm_flow[EVENT_KEYS + [flow_value, flow_event]],
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        lag4_flow,
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )

    lag_balance = f"LAG4_{balance_column}"
    lag_flow = f"LAG4_{flow_value}"
    valid = (
        base[lag_balance].gt(0)
        & base[lag_flow].gt(0)
        & base["LAG4_T_ASSETS"].gt(0)
    )
    balance_growth = base[balance_column].div(base[lag_balance]) - 1.0
    operating_growth = base[flow_value].div(base[lag_flow]) - 1.0
    base[factor] = (operating_growth - balance_growth).where(valid)
    base["EVENT_TIME"] = _latest_time(
        base,
        [
            "BALANCE_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
            flow_event,
            f"LAG4_{flow_event}",
        ],
    )
    base = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[factor, "EVENT_TIME"]
    )
    return base[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            factor,
        ]
    ].copy()


def build_factor_events(
    balance: pd.DataFrame,
    income: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    revenue_ttm = _build_ttm_income(income, "REVENUE")
    cogs_ttm = _build_ttm_income(income, "COGS")
    return {
        "asset_growth": build_asset_growth_events(balance),
        "investment_to_assets": build_investment_events(balance),
        "receivable_abnormal_growth": build_abnormal_growth_events(
            balance,
            revenue_ttm,
            balance_column="AR",
            flow_column="REVENUE",
            factor="receivable_abnormal_growth",
        ),
        "inventory_abnormal_growth": build_abnormal_growth_events(
            balance,
            cogs_ttm,
            balance_column="INVENTORIES",
            flow_column="COGS",
            factor="inventory_abnormal_growth",
        ),
    }


def build_daily_factors(
    events: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    normalized = _normalize_panel(panel).sort_values(KEYS).reset_index(
        drop=True
    )
    result = normalized.copy()
    for factor in FACTOR_COLUMNS:
        one = _carry_one_factor(events[factor], normalized, factor)
        one = _winsorize_daily(one, factor, WINSOR_LIMITS)
        if not one[KEYS].equals(result[KEYS]):
            raise RuntimeError(f"{factor}映射后主键顺序发生变化")
        result[factor] = one[factor].to_numpy()
    return result


def append_factors_atomically(
    factor_path: str | Path,
    factor_values: pd.DataFrame,
) -> tuple[Path, Path]:
    path = Path(factor_path).resolve()
    existing = pd.read_parquet(path)
    existing["TRADE_DATE"] = (
        pd.to_datetime(existing["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    existing["SECURITY_ID"] = pd.to_numeric(
        existing["SECURITY_ID"],
        errors="raise",
    ).astype("int64")
    if existing.duplicated(KEYS).any():
        raise ValueError("原factors.parquet存在重复键")

    values = factor_values[KEYS + FACTOR_COLUMNS].copy()
    if values.duplicated(KEYS).any():
        raise ValueError("第二批四因子结果存在重复键")
    updated = pd.merge(
        existing.drop(columns=FACTOR_COLUMNS, errors="ignore"),
        values,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(updated) != len(existing):
        raise RuntimeError("加入第二批四因子后factors.parquet行数变化")

    backup = path.with_name(
        f"{path.stem}_before_priority_financial_factors_part2{path.suffix}"
    )
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    updated.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    if pq.ParquetFile(temporary).metadata.num_rows != len(existing):
        raise RuntimeError("临时factors.parquet行数校验失败")
    os.replace(temporary, path)
    return path, backup


def run_priority_factors_part2(
    pit_dir: str | Path = DEFAULT_PIT_DIR,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict:
    pit_dir = Path(pit_dir).resolve()
    factor_path = Path(factor_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("读取资产负债表和利润表PIT...")
    balance, income = _read_statements(pit_dir)
    print(f"  balance={len(balance):,}, income={len(income):,}")
    events = build_factor_events(balance, income)
    for factor in FACTOR_COLUMNS:
        print(
            f"  {factor}: events={len(events[factor]):,}, "
            f"stocks={events[factor]['SECURITY_ID'].nunique():,}"
        )

    factor_keys = pd.read_parquet(
        factor_path,
        columns=KEYS,
        engine="pyarrow",
    )
    daily = build_daily_factors(events, factor_keys)
    output_path, backup_path = append_factors_atomically(
        factor_path,
        daily,
    )

    daily_path = (
        audit_dir / "priority_financial_factors_part2_daily.parquet"
    )
    diagnostics_path = (
        audit_dir / "priority_financial_factors_part2_diagnostics.json"
    )
    daily.to_parquet(
        daily_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    definitions = {
        "asset_growth": "1 - total_assets / lag4_total_assets",
        "investment_to_assets": (
            "-(change fixed_assets_total + change inventories) / "
            "lag4 total_assets"
        ),
        "receivable_abnormal_growth": (
            "TTM revenue growth - accounts-receivable growth"
        ),
        "inventory_abnormal_growth": (
            "TTM COGS growth - inventory growth"
        ),
    }
    diagnostics: dict[str, object] = {
        "definitions": definitions,
        "winsor_limits": list(WINSOR_LIMITS),
        "balance_rows": len(balance),
        "income_rows": len(income),
        "daily_panel_rows": len(daily),
        "factor_path": str(output_path),
        "backup_path": str(backup_path),
        "daily_path": str(daily_path),
        "factors": {},
        "event_paths": {},
    }
    for factor in FACTOR_COLUMNS:
        event_path = audit_dir / f"{factor}_events.parquet"
        events[factor].to_parquet(
            event_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        valid = daily[factor].notna()
        diagnostics["event_paths"][factor] = str(event_path)
        diagnostics["factors"][factor] = {
            "event_rows": len(events[factor]),
            "event_stocks": int(events[factor]["SECURITY_ID"].nunique()),
            "event_start": str(events[factor]["EVENT_TIME"].min()),
            "event_end": str(events[factor]["EVENT_TIME"].max()),
            "daily_non_null": int(valid.sum()),
            "daily_coverage": float(valid.mean()),
            "factor_start": str(daily.loc[valid, "TRADE_DATE"].min()),
            "factor_end": str(daily.loc[valid, "TRADE_DATE"].max()),
        }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return {
        "factor_path": output_path,
        "backup_path": backup_path,
        "daily_path": daily_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造第二批四个严格PIT财务因子并加入factors.parquet"
    )
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_priority_factors_part2(
        args.pit_dir,
        args.factors,
        args.audit_dir,
    )


if __name__ == "__main__":
    main()

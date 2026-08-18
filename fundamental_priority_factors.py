"""Build four PIT-safe continuous financial factors.

The script appends these columns to ``factors.parquet``:

``operating_profit_growth``
    (current TTM operating profit - prior-year TTM operating profit) /
    prior-year total assets.

``operating_profit_acceleration``
    Current operating-profit growth minus the immediately preceding fiscal
    quarter's operating-profit growth.

``cfo_sue``
    Seasonal change in standalone-quarter operating cash flow divided by the
    standard deviation of the preceding eight seasonal changes.

``accrual_quality``
    (TTM operating cash flow - TTM consolidated net income) / average total
    assets.  Larger values mean higher cash content and lower accruals.

All flow values are converted from cumulative Chinese financial statements
into standalone fiscal quarters. The first valid PIT observation is used and
each event becomes tradable no earlier than the first market open after
``ACT_PUBTIME``. Financial firms are excluded because their statements are not
comparable with industrial-company statements.
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

from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import (
    BALANCE_COLUMNS,
    COMMON_COLUMNS,
    build_quarterly_balance,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIT_DIR = BASE_DIR / "data" / "new_pit"
DEFAULT_FACTOR_PATH = BASE_DIR / "factors.parquet"
DEFAULT_AUDIT_DIR = BASE_DIR / "factor_components"

KEYS = ["TRADE_DATE", "SECURITY_ID"]
FACTOR_COLUMNS = [
    "operating_profit_growth",
    "operating_profit_acceleration",
    "cfo_sue",
    "accrual_quality",
]
INCOME_VALUE_COLUMNS = ["OPERATE_PROFIT", "N_INCOME"]
CASHFLOW_VALUE_COLUMNS = ["N_CF_OPERATE_A"]
HISTORY_QUARTERS = 8
WINSOR_LIMITS = (0.01, 0.99)


def _latest_time(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the row-wise latest timestamp."""
    return pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in columns],
        axis=1,
    ).max(axis=1)


def _read_statements(
    pit_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=COMMON_COLUMNS + INCOME_VALUE_COLUMNS,
        engine="pyarrow",
    )
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + CASHFLOW_VALUE_COLUMNS,
        engine="pyarrow",
    )
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=BALANCE_COLUMNS,
        engine="pyarrow",
    )
    return income, cashflow, balance


def build_quarterly_flows(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Convert the three cumulative flow fields to standalone quarters."""
    return {
        "OPERATE_PROFIT": build_standalone_quarterly_metric(
            income,
            "OPERATE_PROFIT",
            name="利润表PIT",
        ),
        "N_INCOME": build_standalone_quarterly_metric(
            income,
            "N_INCOME",
            name="利润表PIT",
        ),
        "N_CF_OPERATE_A": build_standalone_quarterly_metric(
            cashflow,
            "N_CF_OPERATE_A",
            name="现金流量表PIT",
        ),
    }


def build_ttm_metric(
    quarterly: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Build strict four-consecutive-quarter TTM values and availability."""
    required = {
        "SECURITY_ID",
        "QUARTER_INDEX",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "END_DATE",
        "EVENT_TIME",
        value_column,
    }
    missing = sorted(required.difference(quarterly.columns))
    if missing:
        raise KeyError(f"{value_column}季度数据缺少字段: {missing}")

    data = quarterly[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            value_column,
        ]
    ].copy()
    data = data.sort_values(["SECURITY_ID", "QUARTER_INDEX"]).reset_index(
        drop=True
    )
    discontinuity = (
        data.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].diff().ne(1)
    )
    data["_SEGMENT"] = discontinuity.groupby(data["SECURITY_ID"]).cumsum()
    grouping = [data["SECURITY_ID"], data["_SEGMENT"]]

    data[f"TTM_{value_column}"] = (
        data.groupby(grouping, sort=False)[value_column]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .reindex(data.index)
    )
    event_time = pd.to_datetime(data["EVENT_TIME"], errors="coerce")
    event_ns = pd.Series(
        event_time.astype("int64").to_numpy(dtype=np.float64),
        index=data.index,
    ).where(event_time.notna())
    latest_ns = (
        event_ns.groupby(grouping, sort=False)
        .rolling(4, min_periods=4)
        .max()
        .reset_index(level=[0, 1], drop=True)
        .reindex(data.index)
    )
    data[f"TTM_{value_column}_EVENT_TIME"] = pd.to_datetime(
        latest_ns,
        unit="ns",
        errors="coerce",
    )
    return (
        data.drop(columns="_SEGMENT")
        .dropna(
            subset=[
                f"TTM_{value_column}",
                f"TTM_{value_column}_EVENT_TIME",
            ]
        )
        .reset_index(drop=True)
    )


def _balance_views(
    balances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = balances[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "T_ASSETS",
            "BALANCE_EVENT_TIME",
        ]
    ].copy()
    lag4 = current.copy()
    lag4["QUARTER_INDEX"] += 4
    lag4 = lag4.rename(
        columns={
            "T_ASSETS": "LAG4_T_ASSETS",
            "BALANCE_EVENT_TIME": "LAG4_BALANCE_EVENT_TIME",
        }
    )
    return current, lag4


def build_operating_profit_events(
    operating_profit: pd.DataFrame,
    balances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build TTM operating-profit growth and its quarter-on-quarter change."""
    ttm = build_ttm_metric(operating_profit, "OPERATE_PROFIT")
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    current, lag4_balance = _balance_views(balances)

    lag4_ttm = ttm[
        keys
        + [
            "TTM_OPERATE_PROFIT",
            "TTM_OPERATE_PROFIT_EVENT_TIME",
        ]
    ].copy()
    lag4_ttm["QUARTER_INDEX"] += 4
    lag4_ttm = lag4_ttm.rename(
        columns={
            "TTM_OPERATE_PROFIT": "LAG4_TTM_OPERATE_PROFIT",
            "TTM_OPERATE_PROFIT_EVENT_TIME": (
                "LAG4_TTM_OPERATE_PROFIT_EVENT_TIME"
            ),
        }
    )
    growth = pd.merge(
        ttm,
        lag4_ttm,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    growth = pd.merge(
        growth,
        lag4_balance,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    valid_assets = growth["LAG4_T_ASSETS"].gt(0)
    growth["operating_profit_growth"] = (
        (
            growth["TTM_OPERATE_PROFIT"]
            - growth["LAG4_TTM_OPERATE_PROFIT"]
        )
        .div(growth["LAG4_T_ASSETS"])
        .where(valid_assets)
    )
    growth["EVENT_TIME"] = _latest_time(
        growth,
        [
            "TTM_OPERATE_PROFIT_EVENT_TIME",
            "LAG4_TTM_OPERATE_PROFIT_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
        ],
    )
    growth = growth.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["operating_profit_growth", "EVENT_TIME"]
    )
    growth_events = growth[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "operating_profit_growth",
        ]
    ].copy()

    previous = growth_events[
        keys + ["operating_profit_growth", "EVENT_TIME"]
    ].copy()
    previous["QUARTER_INDEX"] += 1
    previous = previous.rename(
        columns={
            "operating_profit_growth": "LAG1_OPERATING_PROFIT_GROWTH",
            "EVENT_TIME": "LAG1_GROWTH_EVENT_TIME",
        }
    )
    acceleration = pd.merge(
        growth_events,
        previous,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    acceleration["operating_profit_acceleration"] = (
        acceleration["operating_profit_growth"]
        - acceleration["LAG1_OPERATING_PROFIT_GROWTH"]
    )
    acceleration["EVENT_TIME"] = _latest_time(
        acceleration,
        ["EVENT_TIME", "LAG1_GROWTH_EVENT_TIME"],
    )
    acceleration = acceleration.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["operating_profit_acceleration", "EVENT_TIME"]
    )
    acceleration_events = acceleration[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "operating_profit_acceleration",
        ]
    ].copy()

    # Keep industrial-company membership aligned with the current fiscal
    # quarter. The merge also prevents a stale income event from entering a
    # quarter for which no comparable industrial balance sheet exists.
    eligible = current[
        keys + ["BALANCE_EVENT_TIME"]
    ].drop_duplicates(keys).rename(
        columns={"BALANCE_EVENT_TIME": "CURRENT_BALANCE_EVENT_TIME"}
    )
    growth_events = pd.merge(
        growth_events,
        eligible,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    growth_events["EVENT_TIME"] = _latest_time(
        growth_events,
        ["EVENT_TIME", "CURRENT_BALANCE_EVENT_TIME"],
    )
    acceleration_events = pd.merge(
        acceleration_events,
        eligible,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    acceleration_events["EVENT_TIME"] = _latest_time(
        acceleration_events,
        ["EVENT_TIME", "CURRENT_BALANCE_EVENT_TIME"],
    )
    growth_events = growth_events.drop(columns="CURRENT_BALANCE_EVENT_TIME")
    acceleration_events = acceleration_events.drop(
        columns="CURRENT_BALANCE_EVENT_TIME"
    )
    return growth_events, acceleration_events


def _historical_surprise_std(group: pd.DataFrame) -> pd.Series:
    """Preceding-eight-quarter standard deviation, excluding the current UE."""
    indexed = group.set_index("QUARTER_INDEX").sort_index()
    full_index = pd.RangeIndex(
        int(indexed.index.min()),
        int(indexed.index.max()) + 1,
    )
    full = indexed.reindex(full_index)
    surprise = pd.to_numeric(full["CFO_SURPRISE"], errors="coerce")
    historical_std = surprise.shift(1).rolling(
        HISTORY_QUARTERS,
        min_periods=HISTORY_QUARTERS,
    ).std(ddof=1)

    availability = pd.to_datetime(
        full["SURPRISE_AVAILABLE_TIME"],
        errors="coerce",
    )
    availability_ns = pd.Series(
        availability.astype("int64").to_numpy(dtype=np.float64),
        index=full_index,
    ).where(availability.notna())
    latest_history_ns = availability_ns.shift(1).rolling(
        HISTORY_QUARTERS,
        min_periods=HISTORY_QUARTERS,
    ).max()
    current_time = pd.to_datetime(full["EVENT_TIME"], errors="coerce")
    current_ns = pd.Series(
        current_time.astype("int64").to_numpy(dtype=np.float64),
        index=full_index,
    ).where(current_time.notna())
    historical_std = historical_std.where(latest_history_ns.le(current_ns))
    return group["QUARTER_INDEX"].map(historical_std).astype("float64")


def build_cfo_sue_events(
    cfo: pd.DataFrame,
    balances: pd.DataFrame,
) -> pd.DataFrame:
    """Build standardized seasonal operating-cash-flow surprises."""
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    current = cfo.rename(
        columns={"EVENT_TIME": "CFO_EVENT_TIME"}
    ).copy()
    lag4 = cfo[
        keys + ["N_CF_OPERATE_A", "EVENT_TIME"]
    ].copy()
    lag4["QUARTER_INDEX"] += 4
    lag4 = lag4.rename(
        columns={
            "N_CF_OPERATE_A": "LAG4_N_CF_OPERATE_A",
            "EVENT_TIME": "LAG4_CFO_EVENT_TIME",
        }
    )
    result = pd.merge(
        current,
        lag4,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    result["CFO_SURPRISE"] = (
        result["N_CF_OPERATE_A"] - result["LAG4_N_CF_OPERATE_A"]
    )
    result["SURPRISE_AVAILABLE_TIME"] = _latest_time(
        result,
        ["CFO_EVENT_TIME", "LAG4_CFO_EVENT_TIME"],
    )
    result["EVENT_TIME"] = result["SURPRISE_AVAILABLE_TIME"]
    historical_std = pd.Series(np.nan, index=result.index, dtype="float64")
    for _, group in result.groupby("SECURITY_ID", sort=False):
        historical_std.loc[group.index] = _historical_surprise_std(
            group
        ).to_numpy()
    result["CFO_HIST_STD"] = historical_std
    valid_std = result["CFO_HIST_STD"].gt(0) & np.isfinite(
        result["CFO_HIST_STD"]
    )
    result["cfo_sue"] = result["CFO_SURPRISE"].div(
        result["CFO_HIST_STD"]
    ).where(valid_std)

    eligible = balances[
        keys + ["BALANCE_EVENT_TIME"]
    ].drop_duplicates(keys)
    result = pd.merge(
        result,
        eligible,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    result["EVENT_TIME"] = _latest_time(
        result,
        ["EVENT_TIME", "BALANCE_EVENT_TIME"],
    )
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["cfo_sue", "EVENT_TIME"]
    )
    return result[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "cfo_sue",
        ]
    ].copy()


def build_accrual_quality_events(
    net_income: pd.DataFrame,
    cfo: pd.DataFrame,
    balances: pd.DataFrame,
) -> pd.DataFrame:
    """Build cash-minus-earnings accrual quality scaled by average assets."""
    net_income_ttm = build_ttm_metric(net_income, "N_INCOME")
    cfo_ttm = build_ttm_metric(cfo, "N_CF_OPERATE_A")
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    current_balance, lag4_balance = _balance_views(balances)

    base = pd.merge(
        net_income_ttm[
            [
                "SECURITY_ID",
                "QUARTER_INDEX",
                "FISCAL_YEAR",
                "FISCAL_QUARTER",
                "END_DATE",
                "TTM_N_INCOME",
                "TTM_N_INCOME_EVENT_TIME",
            ]
        ],
        cfo_ttm[
            keys
            + [
                "TTM_N_CF_OPERATE_A",
                "TTM_N_CF_OPERATE_A_EVENT_TIME",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        current_balance,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    base = pd.merge(
        base,
        lag4_balance,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    average_assets = (base["T_ASSETS"] + base["LAG4_T_ASSETS"]) / 2.0
    base["accrual_quality"] = (
        (base["TTM_N_CF_OPERATE_A"] - base["TTM_N_INCOME"])
        .div(average_assets)
        .where(average_assets.gt(0))
    )
    base["EVENT_TIME"] = _latest_time(
        base,
        [
            "TTM_N_INCOME_EVENT_TIME",
            "TTM_N_CF_OPERATE_A_EVENT_TIME",
            "BALANCE_EVENT_TIME",
            "LAG4_BALANCE_EVENT_TIME",
        ],
    )
    base = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["accrual_quality", "EVENT_TIME"]
    )
    return base[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "accrual_quality",
        ]
    ].copy()


def build_factor_events(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    balance: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    flows = build_quarterly_flows(income, cashflow)
    balances = build_quarterly_balance(balance)
    growth, acceleration = build_operating_profit_events(
        flows["OPERATE_PROFIT"],
        balances,
    )
    return {
        "operating_profit_growth": growth,
        "operating_profit_acceleration": acceleration,
        "cfo_sue": build_cfo_sue_events(
            flows["N_CF_OPERATE_A"],
            balances,
        ),
        "accrual_quality": build_accrual_quality_events(
            flows["N_INCOME"],
            flows["N_CF_OPERATE_A"],
            balances,
        ),
    }


def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(KEYS).difference(panel.columns))
    if missing:
        raise KeyError(f"因子面板缺少主键: {missing}")
    result = panel[KEYS].copy()
    result["TRADE_DATE"] = (
        pd.to_datetime(result["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    result["SECURITY_ID"] = pd.to_numeric(
        result["SECURITY_ID"],
        errors="coerce",
    )
    result = result.dropna(subset=KEYS)
    result["SECURITY_ID"] = result["SECURITY_ID"].astype("int64")
    if result.duplicated(KEYS).any():
        raise ValueError("factors.parquet存在重复证券-交易日键")
    return result


def _carry_one_factor(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    factor: str,
) -> pd.DataFrame:
    available = assign_available_trade_date(
        events,
        panel["TRADE_DATE"].unique(),
    )
    event_groups = {
        int(security_id): group.sort_values(
            ["AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        )
        for security_id, group in available.groupby(
            "SECURITY_ID",
            sort=False,
        )
    }
    pieces: list[pd.DataFrame] = []
    for security_id, stock_days in panel.groupby(
        "SECURITY_ID",
        sort=False,
    ):
        left = stock_days.sort_values("TRADE_DATE")
        right = event_groups.get(int(security_id))
        if right is None or right.empty:
            pieces.append(left.assign(**{factor: np.nan}))
            continue
        # A delayed filing for an older quarter must not overwrite a newer
        # quarter that was already available.
        newest_quarter = right["QUARTER_INDEX"].cummax()
        right = right.loc[right["QUARTER_INDEX"].eq(newest_quarter)]
        right = right.drop_duplicates("AVAILABLE_DATE", keep="last")
        joined = pd.merge_asof(
            left,
            right[["AVAILABLE_DATE", factor]],
            left_on="TRADE_DATE",
            right_on="AVAILABLE_DATE",
            direction="backward",
        ).drop(columns="AVAILABLE_DATE")
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(
        drop=True
    )


def _winsorize_daily(
    daily: pd.DataFrame,
    factor: str,
    limits: tuple[float, float] = WINSOR_LIMITS,
) -> pd.DataFrame:
    lower, upper = limits
    if not 0 <= lower < upper <= 1:
        raise ValueError("winsor limits必须满足0 <= lower < upper <= 1")
    quantiles = (
        daily.groupby("TRADE_DATE", sort=False)[factor]
        .quantile([lower, upper])
        .unstack()
    )
    low = daily["TRADE_DATE"].map(quantiles[lower])
    high = daily["TRADE_DATE"].map(quantiles[upper])
    result = daily.copy()
    result[factor] = result[factor].clip(low, high)
    return result


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
        one = _winsorize_daily(one, factor)
        if not one[KEYS].equals(result[KEYS]):
            raise RuntimeError(f"{factor}映射后主键顺序发生变化")
        result[factor] = one[factor].to_numpy()
    return result


def append_factors_atomically(
    factor_path: str | Path,
    factor_values: pd.DataFrame,
) -> tuple[Path, Path]:
    """Append/replace all four columns with a backup and atomic swap."""
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
        raise ValueError("四因子结果存在重复键")
    updated = pd.merge(
        existing.drop(columns=FACTOR_COLUMNS, errors="ignore"),
        values,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(updated) != len(existing):
        raise RuntimeError("加入四因子后factors.parquet行数发生变化")

    backup_dir = path.parent / "输出与测试" / "因子备份"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (
        f"{path.stem}_before_priority_financial_factors{path.suffix}"
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


def run_priority_factors(
    pit_dir: str | Path = DEFAULT_PIT_DIR,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict:
    pit_dir = Path(pit_dir).resolve()
    factor_path = Path(factor_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("读取三张新准则PIT报表...")
    income, cashflow, balance = _read_statements(pit_dir)
    print(
        f"  income={len(income):,}, cashflow={len(cashflow):,}, "
        f"balance={len(balance):,}"
    )
    events = build_factor_events(income, cashflow, balance)
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

    daily_path = audit_dir / "priority_financial_factors_daily.parquet"
    diagnostics_path = (
        audit_dir / "priority_financial_factors_diagnostics.json"
    )
    daily.to_parquet(
        daily_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    event_paths: dict[str, str] = {}
    diagnostics: dict[str, object] = {
        "definitions": {
            "operating_profit_growth": (
                "(TTM operating profit - lag4 TTM operating profit) / "
                "lag4 total assets"
            ),
            "operating_profit_acceleration": (
                "current operating_profit_growth - prior-quarter "
                "operating_profit_growth"
            ),
            "cfo_sue": (
                "standalone-quarter seasonal CFO change / std of preceding "
                "8 seasonal CFO changes"
            ),
            "accrual_quality": (
                "(TTM operating cash flow - TTM consolidated net income) / "
                "average current and lag4 total assets"
            ),
        },
        "winsor_limits": list(WINSOR_LIMITS),
        "income_rows": len(income),
        "cashflow_rows": len(cashflow),
        "balance_rows": len(balance),
        "daily_panel_rows": len(daily),
        "factor_path": str(output_path),
        "backup_path": str(backup_path),
        "daily_path": str(daily_path),
        "factors": {},
    }
    for factor in FACTOR_COLUMNS:
        event_path = audit_dir / f"{factor}_events.parquet"
        events[factor].to_parquet(
            event_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        event_paths[factor] = str(event_path)
        valid = daily[factor].notna()
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
    diagnostics["event_paths"] = event_paths
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return {
        "factor_path": output_path,
        "backup_path": backup_path,
        "daily_path": daily_path,
        "event_paths": event_paths,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造四个严格PIT财务因子并加入factors.parquet"
    )
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_priority_factors(args.pit_dir, args.factors, args.audit_dir)


if __name__ == "__main__":
    main()

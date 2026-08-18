"""Build a PIT-safe quarterly Piotroski F-score.

The original Piotroski score is annual.  This implementation adapts it to
quarterly Chinese financial statements:

* flow variables are converted from cumulative reports into standalone
  quarters;
* change signals compare the same fiscal quarter one year earlier;
* ROA, CFO and turnover use beginning-of-quarter total assets;
* every input must have been published before the score becomes available;
* banks, brokers and insurers are excluded because the accounting ratios are
  not comparable with industrial firms.

The final score is the sum of nine binary signals and therefore ranges from
zero to nine.
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


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIT_DIR = BASE_DIR / "data" / "new_pit"
DEFAULT_FACTOR_PATH = BASE_DIR / "factors.parquet"
DEFAULT_AUDIT_DIR = BASE_DIR / "factor_components"

FACTOR_NAME = "quarterly_f_score"
KEYS = ["TRADE_DATE", "SECURITY_ID"]
REPORT_TYPES = ("A", "Q1", "S1", "Q3")
REPORT_QUARTERS = {"Q1": 1, "S1": 2, "Q3": 3, "A": 4}
FINANCIAL_INDUSTRIES = {"银行业", "证券业", "保险业"}

COMMON_COLUMNS = [
    "ID",
    "SECURITY_ID",
    "ACT_PUBTIME",
    "END_DATE",
    "END_DATE_REP",
    "REPORT_TYPE",
    "FISCAL_PERIOD",
    "MERGED_FLAG",
    "IS_CURRENT_PERIOD",
]
INCOME_VALUE_COLUMNS = ["N_INCOME_ATTR_P", "REVENUE", "COGS"]
CASHFLOW_VALUE_COLUMNS = ["N_CF_OPERATE_A"]
BALANCE_VALUE_COLUMNS = [
    "T_ASSETS",
    "T_CA",
    "T_CL",
    "LT_BORR",
    "BOND_PAYABLE",
    "PAID_IN_CAPITAL",
    "INDUSTRY_CATEGORY",
]
INCOME_COLUMNS = COMMON_COLUMNS + INCOME_VALUE_COLUMNS
CASHFLOW_COLUMNS = COMMON_COLUMNS + CASHFLOW_VALUE_COLUMNS
BALANCE_COLUMNS = COMMON_COLUMNS + BALANCE_VALUE_COLUMNS

SIGNAL_COLUMNS = [
    "F_ROA_POSITIVE",
    "F_CFO_POSITIVE",
    "F_DELTA_ROA",
    "F_ACCRUAL_QUALITY",
    "F_DELTA_LEVERAGE",
    "F_DELTA_LIQUIDITY",
    "F_NO_EQUITY_ISSUANCE",
    "F_DELTA_GROSS_MARGIN",
    "F_DELTA_ASSET_TURNOVER",
]


def _validate_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{name}缺少字段: {missing}")


def _normalize_common(
    frame: pd.DataFrame,
    value_column: str,
    *,
    name: str,
) -> pd.DataFrame:
    """Return the first PIT observation where one statement item is usable."""
    _validate_columns(frame, COMMON_COLUMNS + [value_column], name)
    data = frame[COMMON_COLUMNS + [value_column]].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(
        data["SECURITY_ID"], errors="coerce"
    )
    data["FISCAL_PERIOD"] = pd.to_numeric(
        data["FISCAL_PERIOD"], errors="coerce"
    )
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
    )
    data = data.loc[mask].dropna(
        subset=[
            "SECURITY_ID",
            "ACT_PUBTIME",
            "END_DATE",
            "FISCAL_PERIOD",
            value_column,
        ]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_PERIOD"] = data["FISCAL_PERIOD"].astype("int16")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")

    event_key = [
        "SECURITY_ID",
        "END_DATE",
        "REPORT_TYPE",
        "FISCAL_PERIOD",
    ]
    data = data.sort_values(event_key + ["ACT_PUBTIME", "ID"])
    return data.drop_duplicates(event_key, keep="first").reset_index(drop=True)


def _component(
    reports: pd.DataFrame,
    value_column: str,
    report_type: str,
    fiscal_period: int,
    prefix: str,
) -> pd.DataFrame:
    selected = reports.loc[
        reports["REPORT_TYPE"].eq(report_type)
        & reports["FISCAL_PERIOD"].eq(fiscal_period),
        [
            "SECURITY_ID",
            "FISCAL_YEAR",
            "END_DATE",
            "ACT_PUBTIME",
            value_column,
        ],
    ].copy()
    return selected.rename(
        columns={
            "END_DATE": f"{prefix}_END_DATE",
            "ACT_PUBTIME": f"{prefix}_EVENT_TIME",
            value_column: f"{prefix}_VALUE",
        }
    )


def _latest_time(*values: pd.Series) -> pd.Series:
    return pd.concat(
        [pd.to_datetime(value, errors="coerce") for value in values],
        axis=1,
    ).max(axis=1)


def _quarter_result(
    frame: pd.DataFrame,
    quarter: int,
    end_date: pd.Series,
    event_time: pd.Series,
    values: pd.Series,
    source: str,
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "FISCAL_YEAR"]].copy()
    result["FISCAL_QUARTER"] = np.int8(quarter)
    result["QUARTER_INDEX"] = (
        result["FISCAL_YEAR"].astype("int64") * 4 + quarter
    )
    result["END_DATE"] = pd.to_datetime(end_date, errors="coerce")
    result["EVENT_TIME"] = pd.to_datetime(event_time, errors="coerce")
    result["VALUE"] = pd.to_numeric(values, errors="coerce")
    result["SOURCE"] = source
    return result


def build_standalone_quarterly_metric(
    statement: pd.DataFrame,
    value_column: str,
    *,
    name: str,
) -> pd.DataFrame:
    """Convert one cumulative flow item into standalone fiscal quarters."""
    reports = _normalize_common(
        statement,
        value_column,
        name=name,
    )
    keys = ["SECURITY_ID", "FISCAL_YEAR"]
    q1 = _component(reports, value_column, "Q1", 3, "Q1")
    h1 = _component(reports, value_column, "S1", 6, "H1")
    q3_single = _component(reports, value_column, "Q3", 3, "Q3_SINGLE")
    q3_cumulative = _component(reports, value_column, "Q3", 9, "Q3_CUM")
    annual = _component(reports, value_column, "A", 12, "A")

    pieces: list[pd.DataFrame] = []
    if not q1.empty:
        pieces.append(
            _quarter_result(
                q1,
                1,
                q1["Q1_END_DATE"],
                q1["Q1_EVENT_TIME"],
                q1["Q1_VALUE"],
                "Q1_REPORTED",
            )
        )

    q2 = pd.merge(h1, q1[keys + ["Q1_VALUE", "Q1_EVENT_TIME"]], on=keys)
    if not q2.empty:
        pieces.append(
            _quarter_result(
                q2,
                2,
                q2["H1_END_DATE"],
                _latest_time(q2["H1_EVENT_TIME"], q2["Q1_EVENT_TIME"]),
                q2["H1_VALUE"] - q2["Q1_VALUE"],
                "H1_MINUS_Q1",
            )
        )

    q3_candidates: list[pd.DataFrame] = []
    if not q3_single.empty:
        q3_candidates.append(
            _quarter_result(
                q3_single,
                3,
                q3_single["Q3_SINGLE_END_DATE"],
                q3_single["Q3_SINGLE_EVENT_TIME"],
                q3_single["Q3_SINGLE_VALUE"],
                "Q3_SINGLE_REPORTED",
            )
        )
    q3_derived = pd.merge(
        q3_cumulative,
        h1[keys + ["H1_VALUE", "H1_EVENT_TIME"]],
        on=keys,
    )
    if not q3_derived.empty:
        q3_candidates.append(
            _quarter_result(
                q3_derived,
                3,
                q3_derived["Q3_CUM_END_DATE"],
                _latest_time(
                    q3_derived["Q3_CUM_EVENT_TIME"],
                    q3_derived["H1_EVENT_TIME"],
                ),
                q3_derived["Q3_CUM_VALUE"] - q3_derived["H1_VALUE"],
                "Q3_CUM_MINUS_H1",
            )
        )
    if q3_candidates:
        q3 = pd.concat(q3_candidates, ignore_index=True)
        q3 = q3.dropna(subset=["EVENT_TIME", "VALUE"])
        q3 = q3.sort_values(
            ["SECURITY_ID", "FISCAL_YEAR", "EVENT_TIME", "SOURCE"]
        ).drop_duplicates(keys, keep="first")
        pieces.append(q3)

    q4 = pd.merge(
        annual,
        q3_cumulative[
            keys + ["Q3_CUM_VALUE", "Q3_CUM_EVENT_TIME"]
        ],
        on=keys,
    )
    if not q4.empty:
        pieces.append(
            _quarter_result(
                q4,
                4,
                q4["A_END_DATE"],
                _latest_time(q4["A_EVENT_TIME"], q4["Q3_CUM_EVENT_TIME"]),
                q4["A_VALUE"] - q4["Q3_CUM_VALUE"],
                "A_MINUS_Q3_CUM",
            )
        )

    if not pieces:
        return pd.DataFrame(
            columns=[
                "SECURITY_ID",
                "FISCAL_YEAR",
                "FISCAL_QUARTER",
                "QUARTER_INDEX",
                "END_DATE",
                "EVENT_TIME",
                value_column,
                f"{value_column}_SOURCE",
            ]
        )
    result = pd.concat(pieces, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["END_DATE", "EVENT_TIME", "VALUE"]
    )
    duplicate = result.duplicated(
        ["SECURITY_ID", "QUARTER_INDEX"], keep=False
    )
    if duplicate.any():
        raise ValueError(f"{name}单季度数据存在重复证券-季度键")
    return (
        result.rename(
            columns={
                "VALUE": value_column,
                "SOURCE": f"{value_column}_SOURCE",
            }
        )
        .sort_values(["SECURITY_ID", "QUARTER_INDEX"])
        .reset_index(drop=True)
    )


def build_quarterly_flows(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    """Build aligned standalone-quarter income and operating cash flow."""
    metric_frames: list[pd.DataFrame] = []
    for value_column in INCOME_VALUE_COLUMNS:
        metric_frames.append(
            build_standalone_quarterly_metric(
                income,
                value_column,
                name="利润表PIT",
            )
        )
    metric_frames.append(
        build_standalone_quarterly_metric(
            cashflow,
            "N_CF_OPERATE_A",
            name="现金流量表PIT",
        )
    )

    keys = [
        "SECURITY_ID",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "QUARTER_INDEX",
    ]
    base = metric_frames[0].rename(
        columns={
            "END_DATE": "INCOME_END_DATE",
            "EVENT_TIME": f"{INCOME_VALUE_COLUMNS[0]}_EVENT_TIME",
        }
    )
    keep = keys + [
        "INCOME_END_DATE",
        f"{INCOME_VALUE_COLUMNS[0]}_EVENT_TIME",
        INCOME_VALUE_COLUMNS[0],
        f"{INCOME_VALUE_COLUMNS[0]}_SOURCE",
    ]
    base = base[keep]

    for frame, value_column in zip(
        metric_frames[1:],
        INCOME_VALUE_COLUMNS[1:] + CASHFLOW_VALUE_COLUMNS,
    ):
        other = frame.rename(
            columns={
                "END_DATE": f"{value_column}_END_DATE",
                "EVENT_TIME": f"{value_column}_EVENT_TIME",
            }
        )
        base = pd.merge(
            base,
            other[
                keys
                + [
                    f"{value_column}_END_DATE",
                    f"{value_column}_EVENT_TIME",
                    value_column,
                    f"{value_column}_SOURCE",
                ]
            ],
            on=keys,
            how="inner",
            validate="one_to_one",
        )

    event_columns = [
        f"{column}_EVENT_TIME"
        for column in INCOME_VALUE_COLUMNS + CASHFLOW_VALUE_COLUMNS
    ]
    base["FLOW_EVENT_TIME"] = pd.concat(
        [pd.to_datetime(base[column]) for column in event_columns],
        axis=1,
    ).max(axis=1)
    return base.sort_values(["SECURITY_ID", "QUARTER_INDEX"]).reset_index(
        drop=True
    )


def build_quarterly_balance(balance: pd.DataFrame) -> pd.DataFrame:
    """Return the earliest complete industrial-company balance snapshot."""
    _validate_columns(balance, BALANCE_COLUMNS, "资产负债表PIT")
    data = balance[BALANCE_COLUMNS].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(
        data["SECURITY_ID"], errors="coerce"
    )
    for column in BALANCE_VALUE_COLUMNS:
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
    data = data.loc[mask].copy()
    data["LT_BORR"] = data["LT_BORR"].fillna(0.0)
    data["BOND_PAYABLE"] = data["BOND_PAYABLE"].fillna(0.0)
    data = data.dropna(
        subset=[
            "SECURITY_ID",
            "ACT_PUBTIME",
            "END_DATE",
            "T_ASSETS",
            "T_CA",
            "T_CL",
            "PAID_IN_CAPITAL",
        ]
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
    data["LONG_TERM_DEBT"] = (
        data["LT_BORR"] + data["BOND_PAYABLE"]
    )
    event_key = ["SECURITY_ID", "QUARTER_INDEX"]
    data = data.sort_values(event_key + ["ACT_PUBTIME", "ID"])
    data = data.drop_duplicates(event_key, keep="first")
    return data[
        event_key
        + [
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "ACT_PUBTIME",
            "INDUSTRY_CATEGORY",
            "T_ASSETS",
            "T_CA",
            "T_CL",
            "LONG_TERM_DEBT",
            "PAID_IN_CAPITAL",
        ]
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def build_quarterly_metrics(
    flows: pd.DataFrame,
    balance: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the raw quarterly ratios used by the nine signals."""
    keys = [
        "SECURITY_ID",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "QUARTER_INDEX",
    ]
    base = pd.merge(
        flows,
        balance,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_FLOW", "_BALANCE"),
    )
    previous = balance[
        ["SECURITY_ID", "QUARTER_INDEX", "T_ASSETS", "BALANCE_EVENT_TIME"]
    ].copy()
    previous["QUARTER_INDEX"] += 1
    previous = previous.rename(
        columns={
            "T_ASSETS": "BEGINNING_ASSETS",
            "BALANCE_EVENT_TIME": "BEGINNING_ASSETS_EVENT_TIME",
        }
    )
    base = pd.merge(
        base,
        previous,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    valid_assets = base["BEGINNING_ASSETS"].gt(0)
    valid_current_assets = base["T_ASSETS"].gt(0)
    valid_revenue = base["REVENUE"].gt(0)
    valid_current_liabilities = base["T_CL"].gt(0)

    base["ROA"] = (
        base["N_INCOME_ATTR_P"].div(base["BEGINNING_ASSETS"])
        .where(valid_assets)
    )
    base["CFO"] = (
        base["N_CF_OPERATE_A"].div(base["BEGINNING_ASSETS"])
        .where(valid_assets)
    )
    base["LEVERAGE"] = (
        base["LONG_TERM_DEBT"].div(base["T_ASSETS"])
        .where(valid_current_assets)
    )
    base["CURRENT_RATIO"] = (
        base["T_CA"].div(base["T_CL"]).where(valid_current_liabilities)
    )
    base["GROSS_MARGIN"] = (
        (base["REVENUE"] - base["COGS"])
        .div(base["REVENUE"])
        .where(valid_revenue)
    )
    base["ASSET_TURNOVER"] = (
        base["REVENUE"].div(base["BEGINNING_ASSETS"]).where(valid_assets)
    )
    base["METRIC_EVENT_TIME"] = pd.concat(
        [
            pd.to_datetime(base["FLOW_EVENT_TIME"]),
            pd.to_datetime(base["BALANCE_EVENT_TIME"]),
            pd.to_datetime(base["BEGINNING_ASSETS_EVENT_TIME"]),
        ],
        axis=1,
    ).max(axis=1)
    return base.replace([np.inf, -np.inf], np.nan)


def calculate_f_score_events(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare each quarter with the same quarter one year earlier."""
    compare_columns = [
        "ROA",
        "CFO",
        "LEVERAGE",
        "CURRENT_RATIO",
        "PAID_IN_CAPITAL",
        "GROSS_MARGIN",
        "ASSET_TURNOVER",
        "METRIC_EVENT_TIME",
    ]
    lagged = metrics[
        ["SECURITY_ID", "QUARTER_INDEX"] + compare_columns
    ].copy()
    lagged["QUARTER_INDEX"] += 4
    lagged = lagged.rename(
        columns={column: f"LAG4_{column}" for column in compare_columns}
    )
    result = pd.merge(
        metrics,
        lagged,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    result["EVENT_TIME"] = _latest_time(
        result["METRIC_EVENT_TIME"],
        result["LAG4_METRIC_EVENT_TIME"],
    )

    required = [
        "ROA",
        "CFO",
        "LEVERAGE",
        "CURRENT_RATIO",
        "PAID_IN_CAPITAL",
        "GROSS_MARGIN",
        "ASSET_TURNOVER",
        "LAG4_ROA",
        "LAG4_LEVERAGE",
        "LAG4_CURRENT_RATIO",
        "LAG4_PAID_IN_CAPITAL",
        "LAG4_GROSS_MARGIN",
        "LAG4_ASSET_TURNOVER",
        "EVENT_TIME",
    ]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=required
    )
    result["F_ROA_POSITIVE"] = result["ROA"].gt(0)
    result["F_CFO_POSITIVE"] = result["CFO"].gt(0)
    result["F_DELTA_ROA"] = result["ROA"].gt(result["LAG4_ROA"])
    result["F_ACCRUAL_QUALITY"] = result["CFO"].gt(result["ROA"])
    result["F_DELTA_LEVERAGE"] = result["LEVERAGE"].lt(
        result["LAG4_LEVERAGE"]
    )
    result["F_DELTA_LIQUIDITY"] = result["CURRENT_RATIO"].gt(
        result["LAG4_CURRENT_RATIO"]
    )
    tolerance = np.maximum(result["LAG4_PAID_IN_CAPITAL"].abs(), 1.0) * 1e-8
    result["F_NO_EQUITY_ISSUANCE"] = result["PAID_IN_CAPITAL"].le(
        result["LAG4_PAID_IN_CAPITAL"] + tolerance
    )
    result["F_DELTA_GROSS_MARGIN"] = result["GROSS_MARGIN"].gt(
        result["LAG4_GROSS_MARGIN"]
    )
    result["F_DELTA_ASSET_TURNOVER"] = result["ASSET_TURNOVER"].gt(
        result["LAG4_ASSET_TURNOVER"]
    )
    for column in SIGNAL_COLUMNS:
        result[column] = result[column].astype("int8")
    result["QUARTERLY_F_SCORE"] = (
        result[SIGNAL_COLUMNS].sum(axis=1).astype("int8")
    )
    result = result.sort_values(
        ["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME"]
    )
    return result.drop_duplicates(
        ["SECURITY_ID", "QUARTER_INDEX"], keep="first"
    ).reset_index(drop=True)


def build_f_score_events(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flows = build_quarterly_flows(income, cashflow)
    quarterly_balance = build_quarterly_balance(balance)
    metrics = build_quarterly_metrics(flows, quarterly_balance)
    return calculate_f_score_events(metrics), metrics


def build_daily_f_score(
    events: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Carry the latest disclosed F-score through the daily factor panel."""
    _validate_columns(panel, KEYS, "因子面板")
    daily_panel = panel[KEYS].copy()
    daily_panel["TRADE_DATE"] = (
        pd.to_datetime(daily_panel["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    daily_panel["SECURITY_ID"] = pd.to_numeric(
        daily_panel["SECURITY_ID"], errors="coerce"
    )
    daily_panel = daily_panel.dropna(subset=KEYS)
    daily_panel["SECURITY_ID"] = daily_panel["SECURITY_ID"].astype("int64")
    if daily_panel.duplicated(KEYS).any():
        raise ValueError("factors.parquet存在重复证券-交易日键")

    available = assign_available_trade_date(
        events,
        daily_panel["TRADE_DATE"].unique(),
    )
    event_groups = {
        int(security_id): group.sort_values(
            ["AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        )
        for security_id, group in available.groupby(
            "SECURITY_ID", sort=False
        )
    }
    pieces: list[pd.DataFrame] = []
    for security_id, stock_days in daily_panel.groupby(
        "SECURITY_ID", sort=False
    ):
        left = stock_days.sort_values("TRADE_DATE")
        right = event_groups.get(int(security_id))
        if right is None or right.empty:
            pieces.append(left.assign(**{FACTOR_NAME: np.nan}))
            continue
        # A late record for an old quarter cannot overwrite a newer report.
        latest_quarter = right["QUARTER_INDEX"].cummax()
        right = right.loc[right["QUARTER_INDEX"].eq(latest_quarter)]
        right = right.drop_duplicates("AVAILABLE_DATE", keep="last")
        joined = pd.merge_asof(
            left,
            right[["AVAILABLE_DATE", "QUARTERLY_F_SCORE"]],
            left_on="TRADE_DATE",
            right_on="AVAILABLE_DATE",
            direction="backward",
        ).drop(columns="AVAILABLE_DATE")
        pieces.append(
            joined.rename(columns={"QUARTERLY_F_SCORE": FACTOR_NAME})
        )
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(
        drop=True
    )


def append_factor_atomically(
    factor_path: str | Path,
    factor_values: pd.DataFrame,
) -> tuple[Path, Path]:
    """Append or replace quarterly_f_score with a backup and atomic swap."""
    path = Path(factor_path).resolve()
    existing = pd.read_parquet(path)
    existing["TRADE_DATE"] = (
        pd.to_datetime(existing["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    existing["SECURITY_ID"] = pd.to_numeric(
        existing["SECURITY_ID"], errors="raise"
    ).astype("int64")
    if existing.duplicated(KEYS).any():
        raise ValueError("原factors.parquet存在重复键")

    values = factor_values[KEYS + [FACTOR_NAME]].copy()
    if values.duplicated(KEYS).any():
        raise ValueError("季度F-score结果存在重复键")
    updated = pd.merge(
        existing.drop(columns=FACTOR_NAME, errors="ignore"),
        values,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(updated) != len(existing):
        raise RuntimeError("加入季度F-score后factors.parquet行数发生变化")

    backup_dir = path.parent / "输出与测试" / "因子备份"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (
        f"{path.stem}_before_quarterly_f_score{path.suffix}"
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


def run_quarterly_f_score(
    pit_dir: str | Path = DEFAULT_PIT_DIR,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict:
    pit_dir = Path(pit_dir).resolve()
    factor_path = Path(factor_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("读取三张PIT报表...")
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=INCOME_COLUMNS,
        engine="pyarrow",
    )
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=BALANCE_COLUMNS,
        engine="pyarrow",
    )
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=CASHFLOW_COLUMNS,
        engine="pyarrow",
    )
    print(
        f"  income={len(income):,}, balance={len(balance):,}, "
        f"cashflow={len(cashflow):,}"
    )
    events, metrics = build_f_score_events(income, balance, cashflow)
    print(
        f"  quarterly metrics={len(metrics):,}, "
        f"valid F-score events={len(events):,}"
    )

    factor_keys = pd.read_parquet(
        factor_path,
        columns=KEYS,
        engine="pyarrow",
    )
    daily = build_daily_f_score(events, factor_keys)
    output_path, backup_path = append_factor_atomically(factor_path, daily)

    event_path = audit_dir / "quarterly_f_score_events.parquet"
    metrics_path = audit_dir / "quarterly_f_score_metrics.parquet"
    daily_path = audit_dir / "quarterly_f_score_daily.parquet"
    diagnostics_path = audit_dir / "quarterly_f_score_diagnostics.json"
    events.to_parquet(
        event_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    metrics.to_parquet(
        metrics_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    daily.to_parquet(
        daily_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    valid_daily = daily[FACTOR_NAME].notna()
    diagnostics = {
        "definition": (
            "quarterly Piotroski F-score; standalone-quarter flow values, "
            "same-quarter-year-over-year change signals"
        ),
        "income_rows": len(income),
        "balance_rows": len(balance),
        "cashflow_rows": len(cashflow),
        "quarterly_metric_rows": len(metrics),
        "valid_event_rows": len(events),
        "event_stocks": int(events["SECURITY_ID"].nunique()),
        "event_start": str(events["EVENT_TIME"].min()),
        "event_end": str(events["EVENT_TIME"].max()),
        "daily_panel_rows": len(daily),
        "daily_non_null": int(valid_daily.sum()),
        "daily_coverage": float(valid_daily.mean()),
        "factor_start": str(daily.loc[valid_daily, "TRADE_DATE"].min()),
        "factor_end": str(daily.loc[valid_daily, "TRADE_DATE"].max()),
        "score_distribution": {
            str(int(key)): int(value)
            for key, value in events["QUARTERLY_F_SCORE"]
            .value_counts()
            .sort_index()
            .items()
        },
        "factor_path": str(output_path),
        "backup_path": str(backup_path),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return {
        "factor_path": output_path,
        "backup_path": backup_path,
        "event_path": event_path,
        "metrics_path": metrics_path,
        "daily_path": daily_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造严格PIT的季度Piotroski F-score并加入factors.parquet"
    )
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_quarterly_f_score(args.pit_dir, args.factors, args.audit_dir)


if __name__ == "__main__":
    main()

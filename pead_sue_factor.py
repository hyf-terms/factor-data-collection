"""Build a PIT-safe PEAD/SUE factor and append it to factors.parquet.

Definition
----------
Quarterly earnings are derived from consolidated attributable-parent net
income (``N_INCOME_ATTR_P``):

* Q1 = Q1 cumulative value
* Q2 = half-year cumulative value - Q1 cumulative value
* Q3 = reported single-quarter value, falling back to Q3 cumulative - H1
* Q4 = annual cumulative value - Q3 cumulative value

Unexpected earnings are the year-over-year change in standalone quarterly
earnings. SUE divides that change by the standard deviation of the preceding
eight quarterly year-over-year changes. Only the first valid publication for
each fiscal quarter is used, and availability is based on ``ACT_PUBTIME``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INCOME_PATH = BASE_DIR / "data" / "new_pit" / "new_pit_income"
DEFAULT_FACTOR_PATH = BASE_DIR / "factors.parquet"
DEFAULT_AUDIT_DIR = BASE_DIR / "factor_components"
KEYS = ["TRADE_DATE", "SECURITY_ID"]
FACTOR_NAME = "pead_sue"
REPORT_TYPES = ("A", "Q1", "S1", "Q3")
HISTORY_QUARTERS = 8

INCOME_COLUMNS = [
    "ID",
    "SECURITY_ID",
    "ACT_PUBTIME",
    "END_DATE",
    "END_DATE_REP",
    "REPORT_TYPE",
    "FISCAL_PERIOD",
    "MERGED_FLAG",
    "IS_CURRENT_PERIOD",
    "N_INCOME_ATTR_P",
]


def _normalize_income(income: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(INCOME_COLUMNS).difference(income.columns))
    if missing:
        raise KeyError(f"利润表PIT缺少字段: {missing}")
    data = income[INCOME_COLUMNS].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(
        data["SECURITY_ID"],
        errors="coerce",
    )
    data["FISCAL_PERIOD"] = pd.to_numeric(
        data["FISCAL_PERIOD"],
        errors="coerce",
    )
    data["N_INCOME_ATTR_P"] = pd.to_numeric(
        data["N_INCOME_ATTR_P"],
        errors="coerce",
    )
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
            "N_INCOME_ATTR_P",
        ]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_PERIOD"] = data["FISCAL_PERIOD"].astype("int16")

    # PEAD reacts to the initial earnings announcement. Later database
    # revisions for the same reported component are deliberately ignored.
    event_key = [
        "SECURITY_ID",
        "END_DATE",
        "REPORT_TYPE",
        "FISCAL_PERIOD",
    ]
    data = data.sort_values(event_key + ["ACT_PUBTIME", "ID"])
    data = data.drop_duplicates(event_key, keep="first")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    return data.reset_index(drop=True)


def _component(
    reports: pd.DataFrame,
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
            "N_INCOME_ATTR_P",
        ],
    ].copy()
    return selected.rename(
        columns={
            "END_DATE": f"{prefix}_END_DATE",
            "ACT_PUBTIME": f"{prefix}_EVENT_TIME",
            "N_INCOME_ATTR_P": f"{prefix}_VALUE",
        }
    )


def _quarter_frame(
    frame: pd.DataFrame,
    quarter: int,
    end_date_column: str,
    event_time_column: str,
    earnings: pd.Series,
    source: str | pd.Series,
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "FISCAL_YEAR"]].copy()
    result["FISCAL_QUARTER"] = np.int8(quarter)
    result["QUARTER_INDEX"] = (
        result["FISCAL_YEAR"].astype("int64") * 4 + quarter
    )
    result["END_DATE"] = frame[end_date_column]
    result["EVENT_TIME"] = frame[event_time_column]
    result["QUARTERLY_EARNINGS"] = pd.to_numeric(
        earnings,
        errors="coerce",
    )
    result["SOURCE"] = source
    return result


def build_standalone_quarterly_earnings(
    income: pd.DataFrame,
) -> pd.DataFrame:
    """Convert cumulative PIT income statements into standalone quarters."""
    reports = _normalize_income(income)
    join_keys = ["SECURITY_ID", "FISCAL_YEAR"]
    q1 = _component(reports, "Q1", 3, "Q1")
    h1 = _component(reports, "S1", 6, "H1")
    q3_single = _component(reports, "Q3", 3, "Q3_SINGLE")
    q3_cumulative = _component(reports, "Q3", 9, "Q3_CUM")
    annual = _component(reports, "A", 12, "A")

    quarter_1 = _quarter_frame(
        q1,
        1,
        "Q1_END_DATE",
        "Q1_EVENT_TIME",
        q1["Q1_VALUE"],
        "Q1_REPORTED",
    )

    q2_base = pd.merge(
        h1,
        q1[join_keys + ["Q1_VALUE"]],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )
    quarter_2 = _quarter_frame(
        q2_base,
        2,
        "H1_END_DATE",
        "H1_EVENT_TIME",
        q2_base["H1_VALUE"] - q2_base["Q1_VALUE"],
        "H1_MINUS_Q1",
    )

    q3_base = pd.merge(
        q3_cumulative,
        h1[join_keys + ["H1_VALUE"]],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    q3_base = pd.merge(
        q3_base,
        q3_single[
            join_keys
            + [
                "Q3_SINGLE_VALUE",
                "Q3_SINGLE_END_DATE",
                "Q3_SINGLE_EVENT_TIME",
            ]
        ],
        on=join_keys,
        how="outer",
        validate="one_to_one",
    )
    q3_fallback = q3_base["Q3_CUM_VALUE"] - q3_base["H1_VALUE"]
    q3_earnings = q3_base["Q3_SINGLE_VALUE"].combine_first(q3_fallback)
    q3_end_date = q3_base["Q3_SINGLE_END_DATE"].combine_first(
        q3_base["Q3_CUM_END_DATE"]
    )
    q3_event_time = q3_base["Q3_SINGLE_EVENT_TIME"].combine_first(
        q3_base["Q3_CUM_EVENT_TIME"]
    )
    q3_source = pd.Series(
        np.where(
            q3_base["Q3_SINGLE_VALUE"].notna(),
            "Q3_SINGLE_REPORTED",
            "Q3_CUM_MINUS_H1",
        ),
        index=q3_base.index,
        dtype="string",
    )
    q3_base["Q3_END_DATE"] = q3_end_date
    q3_base["Q3_EVENT_TIME"] = q3_event_time
    quarter_3 = _quarter_frame(
        q3_base,
        3,
        "Q3_END_DATE",
        "Q3_EVENT_TIME",
        q3_earnings,
        q3_source,
    )

    q4_base = pd.merge(
        annual,
        q3_cumulative[join_keys + ["Q3_CUM_VALUE"]],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )
    quarter_4 = _quarter_frame(
        q4_base,
        4,
        "A_END_DATE",
        "A_EVENT_TIME",
        q4_base["A_VALUE"] - q4_base["Q3_CUM_VALUE"],
        "A_MINUS_Q3_CUM",
    )

    result = pd.concat(
        [quarter_1, quarter_2, quarter_3, quarter_4],
        ignore_index=True,
    )
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=["END_DATE", "EVENT_TIME", "QUARTERLY_EARNINGS"]
    )
    duplicate = result.duplicated(
        ["SECURITY_ID", "QUARTER_INDEX"],
        keep=False,
    )
    if duplicate.any():
        examples = result.loc[
            duplicate,
            ["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER"],
        ].head(10)
        raise ValueError(
            "单季度盈余存在重复键，示例: "
            f"{examples.to_dict('records')}"
        )
    return result.sort_values(
        ["SECURITY_ID", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def _historical_ue_std(group: pd.DataFrame) -> pd.Series:
    indexed = group.set_index("QUARTER_INDEX").sort_index()
    full_index = pd.RangeIndex(
        int(indexed.index.min()),
        int(indexed.index.max()) + 1,
    )
    full = indexed.reindex(full_index)
    unexpected = pd.to_numeric(
        full["UNEXPECTED_EARNINGS"],
        errors="coerce",
    )
    historical_std = unexpected.shift(1).rolling(
        HISTORY_QUARTERS,
        min_periods=HISTORY_QUARTERS,
    ).std(ddof=1)

    ue_available = pd.to_datetime(
        full["UE_AVAILABLE_TIME"],
        errors="coerce",
    )
    available_ns = pd.Series(
        ue_available.astype("int64").to_numpy(dtype=np.float64),
        index=full_index,
    ).where(ue_available.notna())
    latest_historical_availability = available_ns.shift(1).rolling(
        HISTORY_QUARTERS,
        min_periods=HISTORY_QUARTERS,
    ).max()
    current_event_ns = pd.Series(
        pd.to_datetime(
            full["EVENT_TIME"],
            errors="coerce",
        ).astype("int64").to_numpy(dtype=np.float64),
        index=full_index,
    ).where(full["EVENT_TIME"].notna())
    historical_std = historical_std.where(
        latest_historical_availability.le(current_event_ns)
    )
    return group["QUARTER_INDEX"].map(historical_std).astype("float64")


def build_sue_events(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Calculate initial-announcement SUE events."""
    current = quarterly.rename(
        columns={"EVENT_TIME": "EARNINGS_EVENT_TIME"}
    ).copy()
    lagged = quarterly[
        [
            "SECURITY_ID",
            "QUARTER_INDEX",
            "QUARTERLY_EARNINGS",
            "EVENT_TIME",
        ]
    ].copy()
    lagged["QUARTER_INDEX"] += 4
    lagged = lagged.rename(
        columns={
            "QUARTERLY_EARNINGS": "EARNINGS_LAG4",
            "EVENT_TIME": "LAG4_EVENT_TIME",
        }
    )
    current = pd.merge(
        current,
        lagged,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )
    current["UNEXPECTED_EARNINGS"] = (
        current["QUARTERLY_EARNINGS"] - current["EARNINGS_LAG4"]
    )
    current["UE_AVAILABLE_TIME"] = pd.concat(
        [
            current["EARNINGS_EVENT_TIME"],
            current["LAG4_EVENT_TIME"],
        ],
        axis=1,
    ).max(axis=1)
    current["EVENT_TIME"] = current["UE_AVAILABLE_TIME"]
    current["UE_HIST_STD"] = (
        current.groupby("SECURITY_ID", group_keys=False)
        .apply(_historical_ue_std, include_groups=False)
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    valid_std = (
        current["UE_HIST_STD"].gt(0)
        & np.isfinite(current["UE_HIST_STD"])
    )
    current["SUE_RAW"] = current["UNEXPECTED_EARNINGS"].div(
        current["UE_HIST_STD"]
    ).where(valid_std)
    current = current.replace([np.inf, -np.inf], np.nan)
    current = current.dropna(subset=["SUE_RAW"]).copy()
    return current.sort_values(
        ["EVENT_TIME", "SECURITY_ID", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def assign_available_trade_date(
    events: pd.DataFrame,
    trading_calendar: pd.Series | np.ndarray,
    market_open: str = "09:30:00",
) -> pd.DataFrame:
    result = events.copy()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(trading_calendar)
    ).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("交易日历为空")
    market_clock = time.fromisoformat(market_open)
    opening_delta = timedelta(
        hours=market_clock.hour,
        minutes=market_clock.minute,
        seconds=market_clock.second,
        microseconds=market_clock.microsecond,
    )
    event_time = pd.to_datetime(result["EVENT_TIME"], errors="coerce")
    event_day = event_time.dt.normalize()
    candidate = event_day.where(
        event_time.le(event_day + opening_delta),
        event_day + timedelta(days=1),
    )
    positions = calendar.searchsorted(candidate.to_numpy(), side="left")
    available = np.full(
        len(result),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    valid = positions < len(calendar)
    available[valid] = calendar.to_numpy()[positions[valid]]
    result["AVAILABLE_DATE"] = pd.to_datetime(available)
    return result.dropna(subset=["AVAILABLE_DATE"])


def build_daily_pead_sue(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Carry the latest announced SUE to the factor panel."""
    required = set(KEYS)
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(f"因子面板缺少键: {missing}")
    daily_panel = panel[KEYS].copy()
    daily_panel["TRADE_DATE"] = (
        pd.to_datetime(daily_panel["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    daily_panel["SECURITY_ID"] = pd.to_numeric(
        daily_panel["SECURITY_ID"],
        errors="coerce",
    )
    daily_panel = daily_panel.dropna(subset=KEYS)
    daily_panel["SECURITY_ID"] = daily_panel["SECURITY_ID"].astype("int64")
    duplicate = daily_panel.duplicated(KEYS, keep=False)
    if duplicate.any():
        raise ValueError(
            f"factors.parquet 存在 {int(duplicate.sum()):,} 行重复键"
        )

    available_events = assign_available_trade_date(
        events,
        daily_panel["TRADE_DATE"].unique(),
    )
    event_groups = {
        int(security_id): group.sort_values(
            ["AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        )
        for security_id, group in available_events.groupby(
            "SECURITY_ID",
            sort=False,
        )
    }

    pieces: list[pd.DataFrame] = []
    for security_id, stock_days in daily_panel.groupby(
        "SECURITY_ID",
        sort=False,
    ):
        left = stock_days.sort_values("TRADE_DATE")
        right = event_groups.get(int(security_id))
        if right is None or right.empty:
            pieces.append(left.assign(**{FACTOR_NAME: np.nan}))
            continue
        # A late disclosure for an older quarter cannot overwrite a newer
        # fiscal quarter that was already public.
        latest_quarter = right["QUARTER_INDEX"].cummax()
        right = right.loc[right["QUARTER_INDEX"].eq(latest_quarter)]
        right = right.drop_duplicates("AVAILABLE_DATE", keep="last")
        joined = pd.merge_asof(
            left,
            right[["AVAILABLE_DATE", "SUE_RAW"]],
            left_on="TRADE_DATE",
            right_on="AVAILABLE_DATE",
            direction="backward",
        ).drop(columns="AVAILABLE_DATE")
        pieces.append(joined.rename(columns={"SUE_RAW": FACTOR_NAME}))

    daily = pd.concat(pieces, ignore_index=True)
    lower, upper = winsor_limits
    if not 0 <= lower < upper <= 1:
        raise ValueError("winsor_limits 必须满足 0 <= lower < upper <= 1")
    quantiles = (
        daily.groupby("TRADE_DATE")[FACTOR_NAME]
        .quantile([lower, upper])
        .unstack()
    )
    low = daily["TRADE_DATE"].map(quantiles[lower])
    high = daily["TRADE_DATE"].map(quantiles[upper])
    daily[FACTOR_NAME] = daily[FACTOR_NAME].clip(low, high)
    return daily.sort_values(KEYS).reset_index(drop=True)


def append_factor_atomically(
    factor_path: str | Path,
    factor_values: pd.DataFrame,
) -> tuple[Path, Path]:
    """Append or replace the PEAD/SUE column with backup and atomic replace."""
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
        raise ValueError("原 factors.parquet 存在重复键")
    values = factor_values[KEYS + [FACTOR_NAME]].copy()
    if values.duplicated(KEYS).any():
        raise ValueError("PEAD/SUE 结果存在重复键")

    base = existing.drop(columns=FACTOR_NAME, errors="ignore")
    updated = pd.merge(
        base,
        values,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(updated) != len(existing):
        raise RuntimeError("合并后 factors.parquet 行数发生变化")

    backup = path.with_name(
        f"{path.stem}_before_pead_sue{path.suffix}"
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
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows != len(existing):
        raise RuntimeError("临时 factors.parquet 行数校验失败")
    os.replace(temporary, path)
    return path, backup


def run_pead_sue(
    income_path: str | Path = DEFAULT_INCOME_PATH,
    factor_path: str | Path = DEFAULT_FACTOR_PATH,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict:
    """Build PEAD/SUE, append it to factors.parquet and save audit data."""
    income_path = Path(income_path).resolve()
    factor_path = Path(factor_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    print("读取利润表PIT...")
    income = pd.read_parquet(
        income_path,
        columns=INCOME_COLUMNS,
        engine="pyarrow",
    )
    print(f"  PIT rows={len(income):,}")
    quarterly = build_standalone_quarterly_earnings(income)
    print(f"  standalone quarters={len(quarterly):,}")
    events = build_sue_events(quarterly)
    print(f"  valid SUE events={len(events):,}")

    factors = pd.read_parquet(
        factor_path,
        columns=KEYS,
        engine="pyarrow",
    )
    daily = build_daily_pead_sue(events, factors)
    output_path, backup_path = append_factor_atomically(
        factor_path,
        daily,
    )

    event_path = audit_dir / "pead_sue_events.parquet"
    daily_path = audit_dir / "pead_sue_daily.parquet"
    diagnostics_path = audit_dir / "pead_sue_diagnostics.json"
    events.to_parquet(
        event_path,
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
    diagnostics = {
        "definition": (
            "quarterly N_INCOME_ATTR_P seasonal difference divided by "
            "the standard deviation of the preceding 8 seasonal differences"
        ),
        "pit_rows": len(income),
        "standalone_quarter_rows": len(quarterly),
        "valid_sue_events": len(events),
        "daily_panel_rows": len(daily),
        "daily_non_null": int(daily[FACTOR_NAME].notna().sum()),
        "daily_coverage": float(daily[FACTOR_NAME].notna().mean()),
        "event_start": str(events["EVENT_TIME"].min()),
        "event_end": str(events["EVENT_TIME"].max()),
        "factor_start": str(
            daily.loc[daily[FACTOR_NAME].notna(), "TRADE_DATE"].min()
        ),
        "factor_end": str(
            daily.loc[daily[FACTOR_NAME].notna(), "TRADE_DATE"].max()
        ),
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
        "daily_path": daily_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造严格PIT的PEAD/SUE并加入factors.parquet"
    )
    parser.add_argument("--income", type=Path, default=DEFAULT_INCOME_PATH)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTOR_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pead_sue(args.income, args.factors, args.audit_dir)


if __name__ == "__main__":
    main()

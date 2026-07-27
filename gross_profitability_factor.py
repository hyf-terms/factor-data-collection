"""Novy-Marx 毛利盈利因子（Gross Profits-to-Assets）的 PIT 实现。

论文原始口径：

    GROSS_PROFITABILITY = (REVENUE - COGS) / T_ASSETS

默认只使用合并年报。程序以 ACT_PUBTIME 判断财报何时可用，并把事件值映射到
交易日股票池；不会按 END_DATE 提前使用尚未公告的财务数据。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["SECURITY_ID", "END_DATE"]
REQUIRED_COMMON = {
    "SECURITY_ID",
    "END_DATE",
    "ACT_PUBTIME",
    "REPORT_TYPE",
    "MERGED_FLAG",
}


def _prepare_statement(
    frame: pd.DataFrame,
    value_columns: Iterable[str],
    *,
    annual_only: bool,
) -> pd.DataFrame:
    """筛出当前期合并报表，并保留每次真实披露/修订事件。"""
    value_columns = list(value_columns)
    required = REQUIRED_COMMON.union(value_columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"财报缺少字段: {missing}")

    data = frame.reset_index() if "ACT_PUBTIME" not in frame.columns else frame.copy()
    data["ACT_PUBTIME"] = pd.to_datetime(data["ACT_PUBTIME"], errors="coerce")
    data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce").dt.normalize()

    mask = data["MERGED_FLAG"].astype(str).eq("1")
    if annual_only:
        mask &= data["REPORT_TYPE"].astype(str).eq("A")
    if "IS_CURRENT_PERIOD" in data.columns:
        mask &= data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
    elif "END_DATE_REP" in data.columns:
        end_date_rep = pd.to_datetime(data["END_DATE_REP"], errors="coerce").dt.normalize()
        mask &= data["END_DATE"].eq(end_date_rep)

    keep = KEY_COLUMNS + ["ACT_PUBTIME"] + value_columns
    if "ID" in data.columns:
        keep.append("ID")
    data = data.loc[mask, keep].dropna(subset=KEY_COLUMNS + ["ACT_PUBTIME"])

    for column in value_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # 同一披露时刻若存在多条记录，以数据库 ID 最大的版本为准。
    sort_columns = KEY_COLUMNS + ["ACT_PUBTIME"]
    if "ID" in data.columns:
        sort_columns.append("ID")
    data = data.sort_values(sort_columns)
    data = data.drop_duplicates(KEY_COLUMNS + ["ACT_PUBTIME"], keep="last")
    return data.drop(columns="ID", errors="ignore")


def build_gross_profitability_events(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    *,
    annual_only: bool = True,
) -> pd.DataFrame:
    """构造财报披露事件级毛利盈利因子。

    利润表或资产负债表任一方发生披露/修订时，先使用该时点已经公开的两张报表
    最新值，再计算因子。因此因子的 EVENT_TIME 等于组成数据中较晚的可得时点。
    """
    income_clean = _prepare_statement(
        income,
        ["REVENUE", "COGS"],
        annual_only=annual_only,
    ).rename(columns={"ACT_PUBTIME": "INCOME_PUBTIME"})
    balance_clean = _prepare_statement(
        balance,
        ["T_ASSETS"],
        annual_only=annual_only,
    ).rename(columns={"ACT_PUBTIME": "BALANCE_PUBTIME"})

    event_times = pd.concat(
        [
            income_clean[KEY_COLUMNS + ["INCOME_PUBTIME"]].rename(
                columns={"INCOME_PUBTIME": "EVENT_TIME"}
            ),
            balance_clean[KEY_COLUMNS + ["BALANCE_PUBTIME"]].rename(
                columns={"BALANCE_PUBTIME": "EVENT_TIME"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    income_groups = {
        keys: group.sort_values("INCOME_PUBTIME")
        for keys, group in income_clean.groupby(KEY_COLUMNS, sort=False)
    }
    balance_groups = {
        keys: group.sort_values("BALANCE_PUBTIME")
        for keys, group in balance_clean.groupby(KEY_COLUMNS, sort=False)
    }

    pieces: list[pd.DataFrame] = []
    for keys, times in event_times.groupby(KEY_COLUMNS, sort=False):
        if keys not in income_groups or keys not in balance_groups:
            continue
        security_id, end_date = keys
        timeline = times.sort_values("EVENT_TIME").copy()
        inc = income_groups[keys]
        bal = balance_groups[keys]

        timeline = pd.merge_asof(
            timeline,
            inc[["INCOME_PUBTIME", "REVENUE", "COGS"]],
            left_on="EVENT_TIME",
            right_on="INCOME_PUBTIME",
            direction="backward",
        )
        timeline = pd.merge_asof(
            timeline,
            bal[["BALANCE_PUBTIME", "T_ASSETS"]],
            left_on="EVENT_TIME",
            right_on="BALANCE_PUBTIME",
            direction="backward",
        )
        timeline["SECURITY_ID"] = security_id
        timeline["END_DATE"] = end_date
        pieces.append(timeline)

    columns = [
        "SECURITY_ID",
        "END_DATE",
        "EVENT_TIME",
        "INCOME_PUBTIME",
        "BALANCE_PUBTIME",
        "REVENUE",
        "COGS",
        "T_ASSETS",
        "GROSS_PROFIT",
        "GROSS_PROFITABILITY",
    ]
    if not pieces:
        return pd.DataFrame(columns=columns)

    result = pd.concat(pieces, ignore_index=True)
    result["GROSS_PROFIT"] = result["REVENUE"] - result["COGS"]
    valid_assets = result["T_ASSETS"].gt(0) & np.isfinite(result["T_ASSETS"])
    result["GROSS_PROFITABILITY"] = (
        result["GROSS_PROFIT"].div(result["T_ASSETS"]).where(valid_assets)
    )
    result = result.dropna(subset=["GROSS_PROFITABILITY"])

    # 同一次修订若没有改变最终因子值，不重复生成事件。
    result = result.sort_values(["SECURITY_ID", "EVENT_TIME", "END_DATE"])
    changed = result.groupby(KEY_COLUMNS)["GROSS_PROFITABILITY"].transform(
        lambda values: values.ne(values.shift())
    )
    return result.loc[changed, columns].reset_index(drop=True)


def assign_available_trade_date(
    events: pd.DataFrame,
    trading_calendar: Iterable[pd.Timestamp | str],
    *,
    market_open: str = "09:30:00",
) -> pd.DataFrame:
    """把披露时间映射到最早可交易日。

    交易日开盘前（默认 09:30，含）披露可在当日使用；否则顺延到下一交易日。
    周末和节假日披露也顺延到下一交易日。
    """
    result = events.copy()
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_calendar))).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("trading_calendar 不能为空")

    event_time = pd.to_datetime(result["EVENT_TIME"], errors="coerce")
    event_day = event_time.dt.normalize()
    cutoff = event_day + pd.to_timedelta(market_open)
    candidate_day = event_day.where(event_time.le(cutoff), event_day + pd.Timedelta(days=1))
    positions = calendar.searchsorted(candidate_day.to_numpy(), side="left")
    available = np.full(
        len(result),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    valid = positions < len(calendar)
    available[valid] = calendar.to_numpy()[positions[valid]]
    result["AVAILABLE_DATE"] = pd.to_datetime(available).astype("datetime64[ns]")
    return result.dropna(subset=["AVAILABLE_DATE"])


def build_daily_gross_profitability(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    annual_only: bool = True,
    winsor_limits: tuple[float, float] | None = (0.01, 0.99),
) -> pd.DataFrame:
    """生成与标签股票池对齐的日频因子面板。

    universe 至少需要 TRADE_DATE、SECURITY_ID。输出既含原始值，也含日截面
    去极值值和百分位排名；排名越大表示毛利盈利能力越高。
    """
    required = {"TRADE_DATE", "SECURITY_ID"}
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise KeyError(f"universe 缺少字段: {missing}")

    panel = universe[["TRADE_DATE", "SECURITY_ID"]].copy()
    panel["TRADE_DATE"] = (
        pd.to_datetime(panel["TRADE_DATE"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    panel["SECURITY_ID"] = pd.to_numeric(panel["SECURITY_ID"], errors="raise").astype("int64")
    panel = panel.dropna(subset=["TRADE_DATE"]).drop_duplicates()

    events = build_gross_profitability_events(
        income,
        balance,
        annual_only=annual_only,
    )
    events = assign_available_trade_date(events, panel["TRADE_DATE"].unique())

    output: list[pd.DataFrame] = []
    for security_id, stock_days in panel.groupby("SECURITY_ID", sort=False):
        left = stock_days.sort_values("TRADE_DATE")
        right = events.loc[
            events["SECURITY_ID"].eq(security_id),
            ["AVAILABLE_DATE", "EVENT_TIME", "END_DATE", "GROSS_PROFITABILITY"],
        ].sort_values(["AVAILABLE_DATE", "EVENT_TIME", "END_DATE"])
        # 较老年报的迟到修订不能覆盖已经公开的较新年报。
        latest_end_date = right["END_DATE"].cummax()
        right = right.loc[right["END_DATE"].eq(latest_end_date)]
        if right.empty:
            joined = left.assign(
                END_DATE=pd.NaT,
                GROSS_PROFITABILITY=np.nan,
            )
        else:
            joined = pd.merge_asof(
                left,
                right,
                left_on="TRADE_DATE",
                right_on="AVAILABLE_DATE",
                direction="backward",
            ).drop(columns=["AVAILABLE_DATE", "EVENT_TIME"])
        output.append(joined)

    daily = pd.concat(output, ignore_index=True)
    raw = daily["GROSS_PROFITABILITY"]
    if winsor_limits is None:
        daily["GROSS_PROFITABILITY_W"] = raw
    else:
        lower, upper = winsor_limits
        if not 0 <= lower < upper <= 1:
            raise ValueError("winsor_limits 必须满足 0 <= lower < upper <= 1")
        quantiles = (
            daily.groupby("TRADE_DATE")["GROSS_PROFITABILITY"]
            .quantile([lower, upper])
            .unstack()
        )
        low_map = daily["TRADE_DATE"].map(quantiles[lower])
        high_map = daily["TRADE_DATE"].map(quantiles[upper])
        daily["GROSS_PROFITABILITY_W"] = raw.clip(low_map, high_map)

    daily["GROSS_PROFITABILITY_RANK"] = daily.groupby("TRADE_DATE")[
        "GROSS_PROFITABILITY_W"
    ].rank(method="average", pct=True)
    return daily.sort_values(["TRADE_DATE", "SECURITY_ID"]).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 PIT 毛利盈利因子 parquet")
    parser.add_argument("--income", required=True, type=Path, help="利润表 parquet")
    parser.add_argument("--balance", required=True, type=Path, help="资产负债表 parquet")
    parser.add_argument("--universe", required=True, type=Path, help="标签/股票池 parquet")
    parser.add_argument("--output", required=True, type=Path, help="输出 parquet")
    parser.add_argument(
        "--all-reports",
        action="store_true",
        help="使用全部报告期（研究扩展口径，不是论文原始年报口径）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_daily_gross_profitability(
        pd.read_parquet(args.income),
        pd.read_parquet(args.balance),
        pd.read_parquet(args.universe, columns=["TRADE_DATE", "SECURITY_ID"]),
        annual_only=not args.all_reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(
        f"已写入 {args.output}: {len(result):,} 行，"
        f"非空因子 {result['GROSS_PROFITABILITY'].notna().sum():,} 行"
    )


if __name__ == "__main__":
    main()

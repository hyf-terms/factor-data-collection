"""
下载新会计准则下的三大财务报表原始披露 PIT 数据。

数据源：
    vw_fdmt_bs_new  资产负债表
    vw_fdmt_is_new  利润表
    vw_fdmt_cf_new  现金流量表

推荐在 history_data.ipynb 已创建 conn 和 lib 后运行：

    from download_new_pit import download_all_new_pit

    download_all_new_pit(
        conn,
        lib,
        start_date="2018-01-01",
        end_date="2026-07-27",
    )

程序默认写入以下新 ArcticDB symbols，不覆盖原有 balance/income/cashflow：
    new_pit_balance
    new_pit_income
    new_pit_cashflow
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PitTable:
    source_table: str
    symbol: str
    chinese_name: str
    event_key: tuple[str, ...]


PIT_TABLES = (
    PitTable(
        source_table="vw_fdmt_bs_new",
        symbol="new_pit_balance",
        chinese_name="新准则资产负债表PIT",
        event_key=(
            "PARTY_ID",
            "END_DATE",
            "ACT_PUBTIME",
            "MERGED_FLAG",
            "END_DATE_REP",
        ),
    ),
    PitTable(
        source_table="vw_fdmt_is_new",
        symbol="new_pit_income",
        chinese_name="新准则利润表PIT",
        event_key=(
            "PARTY_ID",
            "END_DATE",
            "ACT_PUBTIME",
            "MERGED_FLAG",
            "END_DATE_REP",
            "FISCAL_PERIOD",
        ),
    ),
    PitTable(
        source_table="vw_fdmt_cf_new",
        symbol="new_pit_cashflow",
        chinese_name="新准则现金流量表PIT",
        event_key=(
            "PARTY_ID",
            "END_DATE",
            "ACT_PUBTIME",
            "MERGED_FLAG",
            "END_DATE_REP",
            "FISCAL_PERIOD",
        ),
    ),
)


SQL_TEMPLATE = """
SELECT b.SECURITY_ID, a.*
FROM {source_table} a
JOIN md_security b
  ON a.PARTY_ID = b.PARTY_ID
 AND a.TICKER_SYMBOL = b.TICKER_SYMBOL
 AND a.EXCHANGE_CD = b.EXCHANGE_CD
WHERE a.MERGED_FLAG = '1'
  AND a.REPORT_TYPE IN ('A', 'Q1', 'S1', 'Q3')
  AND a.PUBLISH_DATE >= %s
  AND a.PUBLISH_DATE <= %s
  AND (b.LIST_DATE IS NULL OR b.LIST_DATE <= a.PUBLISH_DATE)
  AND (b.DELIST_DATE IS NULL OR b.DELIST_DATE > a.PUBLISH_DATE)
  AND EXISTS (
      SELECT 1
      FROM md_sec_type c
      WHERE c.SECURITY_ID = b.SECURITY_ID
        AND c.TYPE_ID = '101001001001'
        AND (c.INTO_DATE IS NULL OR c.INTO_DATE <= a.PUBLISH_DATE)
        AND (c.OUT_DATE IS NULL OR c.OUT_DATE > a.PUBLISH_DATE)
  )
ORDER BY
    a.PUBLISH_DATE,
    b.SECURITY_ID,
    a.END_DATE_REP,
    a.END_DATE,
    a.FISCAL_PERIOD,
    a.ACT_PUBTIME,
    a.ID
"""


DATE_COLUMNS = (
    "ACT_PUBTIME",
    "PUBLISH_DATE",
    "END_DATE_REP",
    "END_DATE",
    "UPDATE_TIME",
)

METADATA_COLUMNS = {
    "ID",
    "PARTY_ID",
    "SECURITY_ID",
    "TICKER_SYMBOL",
    "EXCHANGE_CD",
    "ACT_PUBTIME",
    "PUBLISH_DATE",
    "END_DATE_REP",
    "END_DATE",
    "REPORT_TYPE",
    "FISCAL_PERIOD",
    "MERGED_FLAG",
    "ACCOUTING_STANDARDS",
    "CURRENCY_CD",
    "INDUSTRY_CATEGORY",
    "UPDATE_TIME",
    "IS_CURRENT_PERIOD",
}


def _year_chunks(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"start_date {start.date()} 晚于 end_date {end.date()}")

    year = start.year
    while year <= end.year:
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        yield chunk_start, chunk_end
        year += 1


def _last_stored_date(lib, symbol: str) -> pd.Timestamp | None:
    if symbol not in set(lib.list_symbols()):
        return None
    tail = lib.tail(symbol, n=1).data
    if tail.empty:
        return None
    return pd.Timestamp(tail.index.max()).normalize()


def _prepare_chunk(df: pd.DataFrame, spec: PitTable) -> pd.DataFrame:
    if df.empty:
        return df

    required = {
        "ID",
        "PARTY_ID",
        "SECURITY_ID",
        "PUBLISH_DATE",
        "ACT_PUBTIME",
        "END_DATE",
        "END_DATE_REP",
        "FISCAL_PERIOD",
        "MERGED_FLAG",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{spec.source_table} 缺少必要字段: {missing}")

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "PUBLISH_DATE",
            "ACT_PUBTIME",
            "END_DATE",
            "END_DATE_REP",
        ]
    ).copy()

    duplicate_mask = df.duplicated(list(spec.event_key), keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, list(spec.event_key)]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{spec.chinese_name} 发现 {int(duplicate_mask.sum())} 条重复事件；"
            f"示例: {examples}"
        )

    df["IS_CURRENT_PERIOD"] = df["END_DATE"].eq(df["END_DATE_REP"])

    for column in ("ID", "PARTY_ID", "SECURITY_ID"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
    df["FISCAL_PERIOD"] = (
        pd.to_numeric(df["FISCAL_PERIOD"], errors="raise").astype("int16")
    )

    value_columns = [
        column for column in df.columns if column not in METADATA_COLUMNS
    ]
    if value_columns:
        df[value_columns] = (
            df[value_columns]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float64")
        )

    sort_columns = [
        "PUBLISH_DATE",
        "SECURITY_ID",
        "END_DATE_REP",
        "END_DATE",
        "FISCAL_PERIOD",
        "ACT_PUBTIME",
        "ID",
    ]
    return df.sort_values(sort_columns).set_index("PUBLISH_DATE")


def _read_chunk(
    conn,
    spec: PitTable,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    sql = SQL_TEMPLATE.format(source_table=spec.source_table)
    raw = pd.read_sql_query(
        sql,
        conn,
        params=(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ),
    )
    return _prepare_chunk(raw, spec)


def download_one_table(
    conn,
    lib,
    spec: PitTable,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp | None = None,
    resume: bool = True,
) -> dict:
    """
    按自然年下载一张 PIT 表并写入 ArcticDB。

    resume=True 时，从目标 symbol 已有的最大 PUBLISH_DATE 的下一日继续，
    因而不会重复 append 已完成区间。
    """
    end = (
        pd.Timestamp.today().normalize()
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    start = pd.Timestamp(start_date).normalize()

    last_date = _last_stored_date(lib, spec.symbol) if resume else None
    if last_date is not None:
        start = max(start, last_date + pd.Timedelta(days=1))

    summary = {
        "source_table": spec.source_table,
        "symbol": spec.symbol,
        "rows": 0,
        "chunks": 0,
        "start_date": start,
        "end_date": end,
    }

    if start > end:
        print(f"{spec.chinese_name}: 已是最新，无需下载")
        return summary

    symbol_exists = spec.symbol in set(lib.list_symbols())
    for chunk_start, chunk_end in _year_chunks(start, end):
        print(
            f"{spec.chinese_name}: "
            f"{chunk_start.date()} 至 {chunk_end.date()} 开始读取"
        )
        data = _read_chunk(conn, spec, chunk_start, chunk_end)
        if data.empty:
            print("  本区间无数据")
            continue

        if symbol_exists:
            lib.append(spec.symbol, data, validate_index=True)
        else:
            lib.write(
                spec.symbol,
                data,
                metadata={
                    "source_table": spec.source_table,
                    "data_type": "raw_disclosure_pit",
                    "accounting_standard": "new",
                    "index": "PUBLISH_DATE",
                },
            )
            symbol_exists = True

        summary["rows"] += len(data)
        summary["chunks"] += 1
        print(
            f"  写入 {len(data):,} 行，"
            f"PUBLISH_DATE 最大值={data.index.max().date()}"
        )

    return summary


def download_all_new_pit(
    conn,
    lib,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """下载新准则资产负债表、利润表、现金流量表三张原始披露 PIT 表。"""
    summaries = [
        download_one_table(
            conn=conn,
            lib=lib,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
            resume=resume,
        )
        for spec in PIT_TABLES
    ]
    result = pd.DataFrame(summaries)
    print("\n下载汇总")
    print(result[["source_table", "symbol", "rows", "chunks"]].to_string(index=False))
    return result


def _connect_from_environment():
    """命令行模式：从环境变量创建数据库和 ArcticDB 连接。"""
    import MySQLdb
    import arcticdb as adb

    required = {
        "PIT_DB_HOST": os.getenv("PIT_DB_HOST"),
        "PIT_DB_USER": os.getenv("PIT_DB_USER"),
        "PIT_DB_PASSWORD": os.getenv("PIT_DB_PASSWORD"),
        "PIT_DB_NAME": os.getenv("PIT_DB_NAME"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"缺少环境变量: {missing}")

    conn = MySQLdb.connect(
        host=required["PIT_DB_HOST"],
        port=int(os.getenv("PIT_DB_PORT", "3306")),
        user=required["PIT_DB_USER"],
        password=required["PIT_DB_PASSWORD"],
        database=required["PIT_DB_NAME"],
        charset=os.getenv("PIT_DB_CHARSET", "utf8mb4"),
    )
    arctic = adb.Arctic(
        os.getenv(
            "PIT_ARCTIC_URI",
            "lmdb://C:/nz/arcticdb?map_size=600GB",
        )
    )
    library_name = os.getenv("PIT_ARCTIC_LIBRARY", "hermes")
    return conn, arctic[library_name]


if __name__ == "__main__":
    database_connection, arctic_library = _connect_from_environment()
    try:
        download_all_new_pit(
            database_connection,
            arctic_library,
            start_date=os.getenv("PIT_START_DATE", "2018-01-01"),
            end_date=os.getenv("PIT_END_DATE") or None,
            resume=True,
        )
    finally:
        database_connection.close()

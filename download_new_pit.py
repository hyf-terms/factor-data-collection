r"""下载新会计准则三大财务报表原始披露 PIT 数据并保存为 Parquet。

数据源：
    vw_fdmt_bs_new  资产负债表
    vw_fdmt_is_new  利润表
    vw_fdmt_cf_new  现金流量表

默认输出：
    本脚本目录/data/new_pit/new_pit_balance/year=YYYY/*.parquet
    本脚本目录/data/new_pit/new_pit_income/year=YYYY/*.parquet
    本脚本目录/data/new_pit/new_pit_cashflow/year=YYYY/*.parquet

推荐在 history_data.ipynb 已创建 conn 后运行：

    from download_new_pit import download_all_new_pit

    summary = download_all_new_pit(
        conn=conn,
        output_dir=r"C:\Users\hyf\Desktop\因子\data\new_pit",
        start_date="2018-01-01",
        end_date="2026-07-27",
        resume=True,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "new_pit"


@dataclass(frozen=True)
class PitTable:
    source_table: str
    dataset_name: str
    chinese_name: str
    event_key: tuple[str, ...]


PIT_TABLES = (
    PitTable(
        source_table="vw_fdmt_bs_new",
        dataset_name="new_pit_balance",
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
        dataset_name="new_pit_income",
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
        dataset_name="new_pit_cashflow",
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

INTEGER_COLUMNS = ("ID", "PARTY_ID", "SECURITY_ID")

TEXT_COLUMNS = (
    "TICKER_SYMBOL",
    "EXCHANGE_CD",
    "REPORT_TYPE",
    "MERGED_FLAG",
    "ACCOUTING_STANDARDS",
    "CURRENCY_CD",
    "INDUSTRY_CATEGORY",
)

METADATA_COLUMNS = {
    *INTEGER_COLUMNS,
    *TEXT_COLUMNS,
    *DATE_COLUMNS,
    "FISCAL_PERIOD",
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

    for year in range(start.year, end.year + 1):
        yield (
            max(start, pd.Timestamp(year=year, month=1, day=1)),
            min(end, pd.Timestamp(year=year, month=12, day=31)),
        )


def _dataset_dir(output_dir: str | Path, spec: PitTable) -> Path:
    return Path(output_dir).expanduser().resolve() / spec.dataset_name


def _parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*.parquet")
        if path.is_file()
    )


def _last_stored_date(dataset_dir: Path) -> pd.Timestamp | None:
    """从已有分区的 PUBLISH_DATE 列确定断点，不依赖文件名猜测。"""
    maxima: list[pd.Timestamp] = []
    for path in _parquet_files(dataset_dir):
        dates = pd.read_parquet(path, columns=["PUBLISH_DATE"])["PUBLISH_DATE"]
        if not dates.empty:
            maxima.append(pd.Timestamp(dates.max()).normalize())
    return max(maxima) if maxima else None


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

    result = df.copy()
    for column in DATE_COLUMNS:
        if column in result.columns:
            result[column] = (
                pd.to_datetime(result[column], errors="coerce")
                .astype("datetime64[ns]")
            )

    result = result.dropna(
        subset=["PUBLISH_DATE", "ACT_PUBTIME", "END_DATE", "END_DATE_REP"]
    ).copy()

    duplicate_mask = result.duplicated(list(spec.event_key), keep=False)
    if duplicate_mask.any():
        examples = (
            result.loc[duplicate_mask, list(spec.event_key)]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{spec.chinese_name} 发现 {int(duplicate_mask.sum())} 条重复事件；"
            f"示例: {examples}"
        )

    result["IS_CURRENT_PERIOD"] = result["END_DATE"].eq(result["END_DATE_REP"])

    for column in INTEGER_COLUMNS:
        if column in result.columns:
            result[column] = (
                pd.to_numeric(result[column], errors="raise").astype("int64")
            )
    result["FISCAL_PERIOD"] = (
        pd.to_numeric(result["FISCAL_PERIOD"], errors="raise").astype("int16")
    )

    for column in TEXT_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("string")

    value_columns = [
        column for column in result.columns if column not in METADATA_COLUMNS
    ]
    if value_columns:
        result[value_columns] = (
            result[value_columns]
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
    return result.sort_values(sort_columns).reset_index(drop=True)


def _read_chunk(
    conn,
    spec: PitTable,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    raw = pd.read_sql_query(
        SQL_TEMPLATE.format(source_table=spec.source_table),
        conn,
        params=(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ),
    )
    return _prepare_chunk(raw, spec)


def _write_partition_atomic(
    data: pd.DataFrame,
    dataset_dir: Path,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
) -> Path:
    year_dir = dataset_dir / f"year={chunk_start.year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    target = year_dir / (
        f"part-{chunk_start:%Y%m%d}-{chunk_end:%Y%m%d}.parquet"
    )
    if target.exists():
        raise FileExistsError(
            f"目标分区已存在: {target}。请使用 resume=True，"
            "或换一个空目录进行完整重建。"
        )

    temporary = target.with_suffix(".tmp")
    data.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    temporary.replace(target)
    return target


def download_one_table(
    conn,
    output_dir: str | Path,
    spec: PitTable,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp | None = None,
    resume: bool = True,
) -> dict:
    """按自然年下载一张PIT表，并写入年度分区Parquet数据集。"""
    end = (
        pd.Timestamp.today().normalize()
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    start = pd.Timestamp(start_date).normalize()
    dataset_dir = _dataset_dir(output_dir, spec)

    last_date = _last_stored_date(dataset_dir) if resume else None
    if last_date is not None:
        start = max(start, last_date + pd.Timedelta(days=1))

    summary = {
        "source_table": spec.source_table,
        "dataset": spec.dataset_name,
        "output_dir": str(dataset_dir),
        "rows": 0,
        "files": 0,
        "start_date": start,
        "end_date": end,
    }
    if start > end:
        print(f"{spec.chinese_name}: 已是最新，无需下载")
        return summary

    for chunk_start, chunk_end in _year_chunks(start, end):
        print(
            f"{spec.chinese_name}: "
            f"{chunk_start.date()} 至 {chunk_end.date()} 开始读取"
        )
        data = _read_chunk(conn, spec, chunk_start, chunk_end)
        if data.empty:
            print("  本区间无数据")
            continue

        path = _write_partition_atomic(
            data,
            dataset_dir,
            chunk_start,
            chunk_end,
        )
        summary["rows"] += len(data)
        summary["files"] += 1
        print(
            f"  写入 {len(data):,} 行至 {path}，"
            f"PUBLISH_DATE 最大值={data['PUBLISH_DATE'].max().date()}"
        )
    return summary


def download_all_new_pit(
    conn,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """下载三张新准则原始披露PIT表，保存为分区Parquet。"""
    summaries = [
        download_one_table(
            conn=conn,
            output_dir=output_dir,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
            resume=resume,
        )
        for spec in PIT_TABLES
    ]
    result = pd.DataFrame(summaries)
    print("\n下载汇总")
    print(
        result[
            ["source_table", "dataset", "rows", "files", "output_dir"]
        ].to_string(index=False)
    )
    return result


def read_pit_dataset(
    dataset: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """读取一个完整的PIT Parquet数据集，供因子程序或Notebook调用。"""
    path = Path(output_dir).expanduser().resolve() / dataset
    if not path.exists():
        raise FileNotFoundError(f"PIT数据集不存在: {path}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def _connect_from_environment():
    """命令行模式：只创建源MySQL连接，结果直接保存为Parquet。"""
    import MySQLdb

    required = {
        "PIT_DB_HOST": os.getenv("PIT_DB_HOST"),
        "PIT_DB_USER": os.getenv("PIT_DB_USER"),
        "PIT_DB_PASSWORD": os.getenv("PIT_DB_PASSWORD"),
        "PIT_DB_NAME": os.getenv("PIT_DB_NAME"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"缺少环境变量: {missing}")

    return MySQLdb.connect(
        host=required["PIT_DB_HOST"],
        port=int(os.getenv("PIT_DB_PORT", "3306")),
        user=required["PIT_DB_USER"],
        password=required["PIT_DB_PASSWORD"],
        database=required["PIT_DB_NAME"],
        charset=os.getenv("PIT_DB_CHARSET", "utf8mb4"),
    )


if __name__ == "__main__":
    database_connection = _connect_from_environment()
    try:
        download_all_new_pit(
            conn=database_connection,
            output_dir=os.getenv("PIT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
            start_date=os.getenv("PIT_START_DATE", "2018-01-01"),
            end_date=os.getenv("PIT_END_DATE") or None,
            resume=True,
        )
    finally:
        database_connection.close()

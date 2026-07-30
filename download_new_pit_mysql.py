"""Export the three new-standard PIT statements from MySQL to Parquet.

Connection settings are imported from ``new_pit_db_local.py``. That local
file is deliberately ignored by Git so credentials cannot be pushed.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from download_new_pit import (
    DEFAULT_OUTPUT_DIR,
    PIT_TABLES,
    PitTable,
    _partition_path,
    _prepare_chunk,
    _write_partition_atomic,
)
from new_pit_db_local import DB_CONFIG


DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = pd.Timestamp.today().normalize()
DEFAULT_CHUNK_DAYS = 31
A_SHARE_TYPE_ID = "101001001001"


def connect_mysql():
    """Open the configured local MySQL connection."""
    import MySQLdb

    return MySQLdb.connect(
        connect_timeout=10,
        read_timeout=300,
        write_timeout=30,
        **DB_CONFIG,
    )


def check_database(conn) -> pd.DataFrame:
    """Return table access and available publication-date ranges."""
    rows: list[dict] = []
    cursor = conn.cursor()
    try:
        for spec in PIT_TABLES:
            cursor.execute(
                f"""
                SELECT MIN(PUBLISH_DATE), MAX(PUBLISH_DATE), COUNT(*)
                FROM `{spec.source_table}`
                """
            )
            minimum, maximum, count = cursor.fetchone()
            rows.append(
                {
                    "table": spec.source_table,
                    "min_publish_date": minimum,
                    "max_publish_date": maximum,
                    "rows": int(count),
                }
            )
    finally:
        cursor.close()
    return pd.DataFrame(rows)


def _date_chunks(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    days: int,
):
    if days < 1:
        raise ValueError("days must be at least 1")
    cursor = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if cursor > end:
        raise ValueError("start_date must not be after end_date")
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _read_mysql_chunk(
    conn,
    spec: PitTable,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Read one publication-date interval with a stable SECURITY_ID."""
    sql = f"""
        SELECT b.SECURITY_ID, a.*
        FROM `{spec.source_table}` AS a
        JOIN `md_security` AS b
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
              FROM `md_sec_type` AS c
              WHERE c.SECURITY_ID = b.SECURITY_ID
                AND c.TYPE_ID = %s
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
    cursor = conn.cursor()
    try:
        cursor.execute(
            sql,
            (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                A_SHARE_TYPE_ID,
            ),
        )
        columns = [item[0] for item in cursor.description]
        raw = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
    finally:
        cursor.close()
    return _prepare_chunk(raw, spec)


def export_table(
    conn,
    spec: PitTable,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    resume: bool = True,
) -> dict:
    """Export one source view as resumable Parquet partitions."""
    dataset_dir = Path(output_dir).expanduser().resolve() / spec.dataset_name
    summary = {
        "table": spec.source_table,
        "dataset": spec.dataset_name,
        "rows": 0,
        "chunks": 0,
        "skipped_chunks": 0,
        "output_dir": str(dataset_dir),
    }
    for chunk_start, chunk_end in _date_chunks(
        start_date,
        end_date,
        chunk_days,
    ):
        target = _partition_path(dataset_dir, chunk_start, chunk_end)
        if resume and target.exists():
            summary["skipped_chunks"] += 1
            continue
        print(
            f"{spec.source_table}: "
            f"{chunk_start.date()} to {chunk_end.date()}"
        )
        data = _read_mysql_chunk(conn, spec, chunk_start, chunk_end)
        if data.empty:
            print("  no data")
            continue
        path = _write_partition_atomic(data, target)
        summary["rows"] += len(data)
        summary["chunks"] += 1
        print(f"  wrote {len(data):,} rows to {path}")
    return summary


def export_all(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    resume: bool = True,
) -> pd.DataFrame:
    """Export all three PIT views and print a summary."""
    conn = connect_mysql()
    try:
        availability = check_database(conn)
        print("Database availability")
        print(availability.to_string(index=False))
        summaries = [
            export_table(
                conn,
                spec,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
                resume=resume,
            )
            for spec in PIT_TABLES
        ]
    finally:
        conn.close()
    result = pd.DataFrame(summaries)
    print("\nExport summary")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    export_all()

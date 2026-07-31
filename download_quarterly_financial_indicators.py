"""Export DataYes single-quarter financial indicators from MySQL.

The legacy ``fdmt_indi_q`` table shown in the data dictionary has no
publication date and is therefore unsafe for point-in-time factor tests.
This downloader uses its new-standard PIT counterpart,
``fdmt_main_data_q_pit``, and writes publication-date Parquet partitions.

Database credentials are imported from ``new_pit_db_local.py``.  That file
is local-only and ignored by Git.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


SOURCE_TABLE = "fdmt_main_data_q_pit"
DATASET_NAME = "quarterly_financial_indicator_pit"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "quarterly_financial_indicators"
)
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = pd.Timestamp.today().normalize()
DEFAULT_CHUNK_DAYS = 92
A_SHARE_TYPE_ID = "101001001001"

DATE_COLUMNS = (
    "PUBLISH_DATE",
    "END_DATE_REP",
    "END_DATE",
    "UPDATE_TIME",
)
INTEGER_COLUMNS = (
    "ID",
    "PARTY_ID",
    "SECURITY_ID",
    "IS_NEW",
    "FISCAL_PERIOD",
)
TEXT_COLUMNS = ("REPORT_TYPE", "MERGED_FLAG")
METADATA_COLUMNS = {
    *DATE_COLUMNS,
    *INTEGER_COLUMNS,
    *TEXT_COLUMNS,
    "IS_CURRENT_PERIOD",
}
EVENT_KEY = (
    "SECURITY_ID",
    "ID",
    "PUBLISH_DATE",
    "END_DATE_REP",
    "END_DATE",
    "REPORT_TYPE",
    "FISCAL_PERIOD",
)


def _load_db_config() -> dict[str, object]:
    try:
        from new_pit_db_local import DB_CONFIG
    except ImportError as error:
        raise RuntimeError(
            "Missing new_pit_db_local.py. Copy "
            "new_pit_db_local.example.py and fill in the local "
            "read-only MySQL connection."
        ) from error
    return DB_CONFIG


def connect_mysql():
    """Open the configured read connection."""
    import MySQLdb

    config = _load_db_config()
    return MySQLdb.connect(
        connect_timeout=10,
        read_timeout=300,
        write_timeout=30,
        **config,
    )


def _date_chunks(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    days: int = DEFAULT_CHUNK_DAYS,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield non-overlapping publication-date intervals."""
    if days < 1:
        raise ValueError("days must be at least 1")
    cursor = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if cursor > end:
        raise ValueError("start_date must not be after end_date")
    while cursor <= end:
        year_end = pd.Timestamp(year=cursor.year, month=12, day=31)
        chunk_end = min(
            cursor + timedelta(days=days - 1),
            year_end,
            end,
        )
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _partition_path(
    dataset_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Path:
    year_dir = dataset_dir / f"year={start_date.year}"
    filename = (
        f"part-{start_date.strftime('%Y%m%d')}-"
        f"{end_date.strftime('%Y%m%d')}.parquet"
    )
    return year_dir / filename


def _write_partition_atomic(data: pd.DataFrame, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    data.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(target)
    return target


def _prepare_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize one database result and validate PIT event uniqueness."""
    if raw.empty:
        return raw
    result = raw.drop_duplicates().copy()
    required = set(EVENT_KEY) | {
        "PARTY_ID",
        "MERGED_FLAG",
        "IS_NEW",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise KeyError(f"{SOURCE_TABLE} is missing required fields: {missing}")

    for column in DATE_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).astype("datetime64[ns]")
    result = result.dropna(
        subset=["PUBLISH_DATE", "END_DATE_REP", "END_DATE"]
    ).copy()

    result["MERGED_FLAG"] = result["MERGED_FLAG"].astype("string")
    result["REPORT_TYPE"] = result["REPORT_TYPE"].astype("string")
    result = result.loc[
        result["MERGED_FLAG"].eq("1")
        & result["REPORT_TYPE"].isin(("A", "Q1", "S1", "Q3"))
    ].copy()

    duplicate = result.duplicated(list(EVENT_KEY), keep=False)
    if duplicate.any():
        examples = (
            result.loc[duplicate, list(EVENT_KEY)]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{SOURCE_TABLE} has {int(duplicate.sum())} duplicate "
            f"PIT events after the security join; examples: {examples}"
        )

    result["IS_CURRENT_PERIOD"] = result["END_DATE"].eq(
        result["END_DATE_REP"]
    )
    for column in INTEGER_COLUMNS:
        if column not in result:
            continue
        numeric = pd.to_numeric(result[column], errors="raise")
        if numeric.isna().any():
            result[column] = numeric.astype("Int64")
        else:
            result[column] = numeric.astype("int64")
    if "FISCAL_PERIOD" in result:
        result["FISCAL_PERIOD"] = result["FISCAL_PERIOD"].astype("Int16")
    if "IS_NEW" in result:
        result["IS_NEW"] = result["IS_NEW"].astype("Int8")
    for column in TEXT_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")

    values = [
        column for column in result if column not in METADATA_COLUMNS
    ]
    if values:
        result[values] = result[values].apply(
            pd.to_numeric,
            errors="coerce",
        ).astype("float64")

    return result.sort_values(
        [
            "PUBLISH_DATE",
            "SECURITY_ID",
            "END_DATE_REP",
            "END_DATE",
            "FISCAL_PERIOD",
            "ID",
        ]
    ).reset_index(drop=True)


def _read_mysql_chunk(
    connection,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Read one publication-date range and attach the A-share security ID."""
    sql = f"""
        SELECT b.SECURITY_ID, a.*
        FROM `{SOURCE_TABLE}` AS a
        JOIN `md_security` AS b
          ON a.PARTY_ID = b.PARTY_ID
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
          a.ID
    """
    cursor = connection.cursor()
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
    return _prepare_chunk(raw)


def check_database(connection) -> dict[str, object]:
    """Return source availability without exposing connection settings."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"""
            SELECT MIN(PUBLISH_DATE), MAX(PUBLISH_DATE), COUNT(*)
            FROM `{SOURCE_TABLE}`
            WHERE MERGED_FLAG = '1'
              AND REPORT_TYPE IN ('A', 'Q1', 'S1', 'Q3')
            """
        )
        minimum, maximum, rows = cursor.fetchone()
    finally:
        cursor.close()
    return {
        "source_table": SOURCE_TABLE,
        "min_publish_date": minimum,
        "max_publish_date": maximum,
        "source_rows": int(rows),
    }


def write_field_catalog(connection, output_dir: Path) -> Path:
    """Save the database field dictionary beside, not inside, the dataset."""
    config = _load_db_config()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT ORDINAL_POSITION, COLUMN_NAME, COLUMN_TYPE,
                   IS_NULLABLE, COALESCE(COLUMN_COMMENT, '')
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
            """,
            (config["database"], SOURCE_TABLE),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
    catalog = pd.DataFrame(
        rows,
        columns=[
            "ordinal_position",
            "field",
            "mysql_type",
            "nullable",
            "description",
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "quarterly_financial_indicator_fields.csv"
    catalog.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def export_quarterly_indicators(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    resume: bool = True,
    check_only: bool = False,
) -> dict[str, object]:
    """Export the strict-PIT single-quarter indicators to local Parquet."""
    root = Path(output_dir).expanduser().resolve()
    dataset_dir = root / DATASET_NAME
    connection = connect_mysql()
    try:
        availability = check_database(connection)
        print("Database availability")
        print(pd.Series(availability).to_string())
        catalog_path = write_field_catalog(connection, root)
        print(f"Field catalog: {catalog_path}")
        if check_only:
            return {
                **availability,
                "rows": 0,
                "chunks": 0,
                "skipped_chunks": 0,
                "output_dir": str(dataset_dir),
                "field_catalog": str(catalog_path),
            }

        summary: dict[str, object] = {
            **availability,
            "rows": 0,
            "chunks": 0,
            "skipped_chunks": 0,
            "output_dir": str(dataset_dir),
            "field_catalog": str(catalog_path),
        }
        for chunk_start, chunk_end in _date_chunks(
            start_date,
            end_date,
            chunk_days,
        ):
            target = _partition_path(
                dataset_dir,
                chunk_start,
                chunk_end,
            )
            if resume and target.exists():
                summary["skipped_chunks"] = (
                    int(summary["skipped_chunks"]) + 1
                )
                continue
            print(
                f"{SOURCE_TABLE}: publication dates "
                f"{chunk_start.date()} to {chunk_end.date()}"
            )
            data = _read_mysql_chunk(
                connection,
                chunk_start,
                chunk_end,
            )
            if data.empty:
                print("  no data")
                continue
            path = _write_partition_atomic(data, target)
            summary["rows"] = int(summary["rows"]) + len(data)
            summary["chunks"] = int(summary["chunks"]) + 1
            print(f"  wrote {len(data):,} rows to {path}")
    finally:
        connection.close()
    print("\nExport summary")
    print(pd.Series(summary).to_string())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download DataYes new-standard single-quarter financial "
            "indicator PIT data to partitioned Parquet."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE.strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="overwrite existing publication-date partitions",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only check coverage and write the field dictionary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_quarterly_indicators(
        output_dir=arguments.output_dir,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        chunk_days=arguments.chunk_days,
        resume=not arguments.no_resume,
        check_only=arguments.check_only,
    )

"""Export one approved MySQL table to date-partitioned Parquet files.

This is intentionally generic: run the metadata discovery first, verify the
table's economic meaning and PIT/publication-date field, then pass both names
explicitly.  The script never stores credentials in this repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def safe_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return f"`{value}`"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--date-column", required=True)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--columns", nargs="+",
        help="optional explicit column whitelist; identifiers are validated",
    )
    parser.add_argument("--chunk-days", type=int, default=92)
    parser.add_argument("--sql-chunksize", type=int, default=100_000)
    args = parser.parse_args()

    table = safe_identifier(args.table)
    date_column = safe_identifier(args.date_column)
    selected_columns = [safe_identifier(column) for column in args.columns] if args.columns else []
    if selected_columns and date_column not in selected_columns:
        selected_columns.append(date_column)
    select_clause = ", ".join(selected_columns) if selected_columns else "*"
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    if end < start or args.chunk_days < 1:
        raise ValueError("invalid date range or chunk size")

    sys.path.insert(0, str(args.config_dir.resolve()))
    from download_new_pit_mysql import connect_mysql

    output = args.output_dir.resolve() / args.table
    output.mkdir(parents=True, exist_ok=True)
    connection = connect_mysql()
    files: list[dict[str, object]] = []
    try:
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + pd.Timedelta(args.chunk_days - 1, unit="D"), end)
            target = output / f"{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.parquet"
            if target.exists():
                files.append({"file": target.name, "status": "existing"})
                cursor = chunk_end + pd.Timedelta(1, unit="D")
                continue
            sql = (
                f"SELECT {select_clause} FROM {table} WHERE {date_column} >= %s "
                f"AND {date_column} < %s ORDER BY {date_column}"
            )
            frames = list(pd.read_sql(
                sql,
                connection,
                params=[cursor.strftime("%Y-%m-%d"), (chunk_end + pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d")],
                chunksize=args.sql_chunksize,
            ))
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            frame.to_parquet(target, index=False)
            files.append({"file": target.name, "status": "written", "rows": len(frame)})
            print(f"{target.name}: {len(frame):,} rows")
            cursor = chunk_end + pd.Timedelta(1, unit="D")
    finally:
        connection.close()

    manifest = {
        "table": args.table,
        "date_column": args.date_column,
        "columns": args.columns,
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

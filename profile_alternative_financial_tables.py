"""Profile row counts and date coverage of selected accessible tables."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
DATE_PRIORITY = [
    "ACT_PUBTIME", "PUBLISH_DATE", "FIRST_PUBLISH_DATE", "PRE_PUB_DATE",
    "REP_FORE_DATE", "STAT_DATE", "THIS_WRITE_DATE", "END_DATE", "UPDATE_TIME",
]


def quote(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(value)
    return f"`{value}`"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("tables", nargs="+")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    sys.path.insert(0, str(args.config_dir.resolve()))
    from download_new_pit_mysql import connect_mysql

    connection = connect_mysql()
    rows: list[dict[str, object]] = []
    try:
        for table in args.tables:
            cols = metadata.loc[metadata["TABLE_NAME"].eq(table), "COLUMN_NAME"].tolist()
            date_col = next((name for name in DATE_PRIORITY if name in cols), None)
            record: dict[str, object] = {"TABLE_NAME": table, "DATE_COLUMN": date_col}
            try:
                if date_col:
                    sql = (
                        f"SELECT COUNT(*) AS ROW_COUNT, MIN({quote(date_col)}) AS MIN_DATE, "
                        f"MAX({quote(date_col)}) AS MAX_DATE, "
                        f"SUM({quote(date_col)} IS NULL) AS NULL_DATES FROM {quote(table)}"
                    )
                else:
                    sql = f"SELECT COUNT(*) AS ROW_COUNT FROM {quote(table)}"
                profile = pd.read_sql(sql, connection).iloc[0].to_dict()
                record.update(profile)
                record["STATUS"] = "accessible"
            except Exception as error:  # record table-level permission failures
                record["STATUS"] = f"error: {type(error).__name__}: {error}"
            rows.append(record)
            print(record)
    finally:
        connection.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

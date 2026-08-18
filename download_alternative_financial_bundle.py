"""Download the approved alternative-financial bundle to yearly Parquet.

The selected sources cover analyst revisions, preannouncements/express reports,
audits, accounting-note details, major contracts and ownership actions.  Daily
consensus snapshots are deliberately excluded because the accessible table is
very large; ``rr_profit_adjust_v2`` contains the revision event itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from download_mysql_table_parquet import safe_identifier


SOURCES = {
    "rr_profit_adjust_v2": "THIS_WRITE_DATE",
    "fdmt_ef_v2": "ACT_PUBTIME",
    "fdmt_ee": "ACT_PUBTIME",
    "fdmt_adt_opn_n": "PUBLISH_DATE",
    "fdmt_main_adt_matters": "PUBLISH_DATE",
    "equ_major_contract_pit": "PUBLISH_DATE",
    "equ_change_plan": "PUBLISH_DATE",
    "equ_share_buy_back": "PUBLISH_DATE",
    "fdmt_acc_rec_age": "ACT_PUBTIME",
    "fdmt_oth_rec_age": "ACT_PUBTIME",
    "fdmt_ass_imp_lossv2": "ACT_PUBTIME",
    "fdmt_ass_imp_pre": "ACT_PUBTIME",
    "fdmt_con_lab_na": "ACT_PUBTIME",
}


def write_query(connection, sql: str, params: list[str], target: Path, chunksize: int) -> int:
    temporary = target.with_suffix(".parquet.partial")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for frame in pd.read_sql(sql, connection, params=params, chunksize=chunksize):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
        if writer is not None:
            writer.close()
            writer = None
        else:
            pd.DataFrame().to_parquet(temporary, index=False)
        temporary.replace(target)
        return rows
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sql-chunksize", type=int, default=100_000)
    args = parser.parse_args()

    end = pd.Timestamp(args.end_date).normalize()
    if args.start_year > end.year:
        raise ValueError("start year is after end date")
    sys.path.insert(0, str(args.config_dir.resolve()))
    from download_new_pit_mysql import connect_mysql

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    connection = connect_mysql()
    try:
        for table_name, date_name in SOURCES.items():
            table_dir = root / table_name
            table_dir.mkdir(exist_ok=True)
            for year in range(args.start_year, end.year + 1):
                lower = pd.Timestamp(year=year, month=1, day=1)
                upper = min(
                    pd.Timestamp(year=year + 1, month=1, day=1),
                    end + pd.Timedelta(1, unit="D"),
                )
                target = table_dir / f"year={year}.parquet"
                if target.exists():
                    rows = pq.ParquetFile(target).metadata.num_rows
                    status = "existing"
                else:
                    sql = (
                        f"SELECT * FROM {safe_identifier(table_name)} "
                        f"WHERE {safe_identifier(date_name)} >= %s "
                        f"AND {safe_identifier(date_name)} < %s "
                        f"ORDER BY {safe_identifier(date_name)}"
                    )
                    rows = write_query(
                        connection,
                        sql,
                        [lower.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")],
                        target,
                        args.sql_chunksize,
                    )
                    status = "written"
                record = {
                    "table": table_name, "date_column": date_name,
                    "year": year, "rows": rows, "status": status,
                    "file": str(target),
                }
                summary.append(record)
                print(f"{table_name} {year}: {rows:,} ({status})", flush=True)
    finally:
        connection.close()

    frame = pd.DataFrame(summary)
    frame.to_csv(root / "download_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "start_year": args.start_year,
        "end_date": str(end.date()),
        "sources": SOURCES,
        "excluded": {
            "con_sec_fy12_2": "16.7m-row daily snapshot; revision-event table used instead",
            "segment_revenue": "no matching accessible table identified",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

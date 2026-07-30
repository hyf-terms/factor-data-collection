"""Download the local inputs required to reproduce the CH-3 and CH-4 models.

The downloader uses the same read-only MySQL configuration as
``download_new_pit_mysql.py`` and writes resumable Parquet partitions.  It
does not write anything back to MySQL.

Datasets
--------
market_daily
    A-share daily prices, adjusted prices, volume, A-share market value and
    listing date.
earnings_pit
    Point-in-time net income excluding non-recurring gains/losses.
share_changes
    Point-in-time total share counts, including A/B/H shares.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from download_new_pit_mysql import A_SHARE_TYPE_ID, connect_mysql


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "ch_models"
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = pd.Timestamp.today().normalize()


@dataclass(frozen=True)
class DateChunk:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def year(self) -> int:
        return int(self.start.year)

    @property
    def stem(self) -> str:
        return f"part-{self.start:%Y%m%d}-{self.end:%Y%m%d}.parquet"


def month_chunks(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> Iterable[DateChunk]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.offsets.MonthEnd(0), end)
        yield DateChunk(cursor, pd.Timestamp(chunk_end))
        cursor = pd.Timestamp(chunk_end) + timedelta(days=1)


def _target_path(root: Path, dataset: str, chunk: DateChunk) -> Path:
    return root / dataset / f"year={chunk.year}" / chunk.stem


def _atomic_parquet(data: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    data.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    os.replace(temporary, target)


def _read_query(conn, sql: str, params: tuple) -> pd.DataFrame:
    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)
        names = [description[0] for description in cursor.description]
        return pd.DataFrame.from_records(cursor.fetchall(), columns=names)
    finally:
        cursor.close()


def _normalize_market(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.copy()
    for column in ("TRADE_DATE", "LIST_DATE", "DELIST_DATE"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["SECURITY_ID"] = pd.to_numeric(
        result["SECURITY_ID"], errors="raise"
    ).astype("int64")
    numeric = [
        "CLOSE_PRICE",
        "ADJ_CLOSE_PRICE",
        "TURNOVER_VOL",
        "MARKET_VALUE_A",
        "CHG_PCT",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.drop_duplicates(
        ["TRADE_DATE", "SECURITY_ID"], keep="last"
    )
    return result.sort_values(["TRADE_DATE", "SECURITY_ID"]).reset_index(
        drop=True
    )


MARKET_SQL = f"""
    SELECT
        a.TRADE_DATE,
        a.SECURITY_ID,
        a.TICKER_SYMBOL,
        a.EXCHANGE_CD,
        b.LIST_DATE,
        b.DELIST_DATE,
        a.CLOSE_PRICE,
        af.CLOSE_PRICE_2 AS ADJ_CLOSE_PRICE,
        a.TURNOVER_VOL,
        a.MARKET_VALUE AS MARKET_VALUE_A,
        a.CHG_PCT
    FROM mkt_equd AS a
    JOIN mkt_equd_adj_af AS af
      ON af.SECURITY_ID = a.SECURITY_ID
     AND af.TRADE_DATE = a.TRADE_DATE
    JOIN md_security AS b
      ON b.SECURITY_ID = a.SECURITY_ID
    WHERE a.TRADE_DATE >= %s
      AND a.TRADE_DATE <= %s
      AND (b.LIST_DATE IS NULL OR b.LIST_DATE <= a.TRADE_DATE)
      AND (b.DELIST_DATE IS NULL OR b.DELIST_DATE >= a.TRADE_DATE)
      AND EXISTS (
          SELECT 1
          FROM md_sec_type AS st
          WHERE st.SECURITY_ID = a.SECURITY_ID
            AND st.TYPE_ID = '{A_SHARE_TYPE_ID}'
            AND (st.INTO_DATE IS NULL OR st.INTO_DATE <= a.TRADE_DATE)
            AND (st.OUT_DATE IS NULL OR st.OUT_DATE > a.TRADE_DATE)
      )
    ORDER BY a.TRADE_DATE, a.SECURITY_ID
"""


EARNINGS_SQL = f"""
    SELECT
        b.SECURITY_ID,
        a.ID,
        a.PARTY_ID,
        a.TICKER_SYMBOL,
        a.EXCHANGE_CD,
        a.ACT_PUBTIME,
        a.PUBLISH_DATE,
        a.END_DATE,
        a.END_DATE_REP,
        a.FISCAL_PERIOD,
        a.REPORT_TYPE,
        a.MERGED_FLAG,
        a.ADJUSTED_FLAG,
        a.NR_PROFIT_LOSS,
        a.N_INCOME_CUT
    FROM fdmt_main_indi_pit AS a
    JOIN md_security AS b
      ON b.PARTY_ID = a.PARTY_ID
     AND b.TICKER_SYMBOL = a.TICKER_SYMBOL
     AND b.EXCHANGE_CD = a.EXCHANGE_CD
    WHERE a.PUBLISH_DATE >= %s
      AND a.PUBLISH_DATE <= %s
      AND a.MERGED_FLAG = '1'
      AND a.REPORT_TYPE IN ('A', 'Q1', 'S1', 'Q3')
      AND a.END_DATE = a.END_DATE_REP
      AND a.ADJUSTED_FLAG LIKE '合并本期%%'
      AND a.N_INCOME_CUT IS NOT NULL
      AND EXISTS (
          SELECT 1
          FROM md_sec_type AS st
          WHERE st.SECURITY_ID = b.SECURITY_ID
            AND st.TYPE_ID = '{A_SHARE_TYPE_ID}'
            AND (
                st.INTO_DATE IS NULL
                OR st.INTO_DATE <= a.PUBLISH_DATE
            )
            AND (
                st.OUT_DATE IS NULL
                OR st.OUT_DATE > a.PUBLISH_DATE
            )
      )
    ORDER BY
        a.PUBLISH_DATE,
        b.SECURITY_ID,
        a.ACT_PUBTIME,
        a.ID
"""


SHARES_SQL = f"""
    SELECT
        b.SECURITY_ID,
        a.ID,
        a.PARTY_ID,
        a.PUBLISH_DATE,
        a.CHANGE_DATE,
        a.A_SHARES,
        a.B_SHARES,
        a.H_SHARES,
        a.TOTAL_SHARES
    FROM equ_share_change AS a
    JOIN md_security AS b
      ON b.PARTY_ID = a.PARTY_ID
    WHERE a.TOTAL_SHARES IS NOT NULL
      AND a.PUBLISH_DATE <= %s
      AND EXISTS (
          SELECT 1
          FROM md_sec_type AS st
          WHERE st.SECURITY_ID = b.SECURITY_ID
            AND st.TYPE_ID = '{A_SHARE_TYPE_ID}'
      )
    ORDER BY
        b.SECURITY_ID,
        a.PUBLISH_DATE,
        a.CHANGE_DATE,
        a.ID
"""


def export_market(
    conn,
    output_dir: Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    resume: bool,
) -> list[dict]:
    summaries: list[dict] = []
    for chunk in month_chunks(start_date, end_date):
        target = _target_path(output_dir, "market_daily", chunk)
        if resume and target.exists():
            summaries.append(
                {
                    "dataset": "market_daily",
                    "start": str(chunk.start.date()),
                    "end": str(chunk.end.date()),
                    "rows": None,
                    "status": "skipped",
                    "path": str(target),
                }
            )
            continue
        print(f"market_daily: {chunk.start.date()} to {chunk.end.date()}")
        data = _read_query(
            conn,
            MARKET_SQL,
            (
                chunk.start.strftime("%Y-%m-%d"),
                chunk.end.strftime("%Y-%m-%d"),
            ),
        )
        data = _normalize_market(data)
        if not data.empty:
            _atomic_parquet(data, target)
        summaries.append(
            {
                "dataset": "market_daily",
                "start": str(chunk.start.date()),
                "end": str(chunk.end.date()),
                "rows": len(data),
                "status": "written" if not data.empty else "empty",
                "path": str(target),
            }
        )
        print(f"  rows: {len(data):,}")
    return summaries


def _normalize_earnings(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.copy()
    for column in (
        "ACT_PUBTIME",
        "PUBLISH_DATE",
        "END_DATE",
        "END_DATE_REP",
    ):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["SECURITY_ID"] = pd.to_numeric(
        result["SECURITY_ID"], errors="raise"
    ).astype("int64")
    for column in ("NR_PROFIT_LOSS", "N_INCOME_CUT"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.drop_duplicates(
        ["SECURITY_ID", "ACT_PUBTIME", "END_DATE"], keep="last"
    )
    return result.sort_values(
        ["PUBLISH_DATE", "SECURITY_ID", "ACT_PUBTIME", "ID"]
    ).reset_index(drop=True)


def export_earnings(
    conn,
    output_dir: Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    resume: bool,
) -> list[dict]:
    summaries: list[dict] = []
    for chunk in month_chunks(start_date, end_date):
        target = _target_path(output_dir, "earnings_pit", chunk)
        if resume and target.exists():
            summaries.append(
                {
                    "dataset": "earnings_pit",
                    "start": str(chunk.start.date()),
                    "end": str(chunk.end.date()),
                    "rows": None,
                    "status": "skipped",
                    "path": str(target),
                }
            )
            continue
        print(f"earnings_pit: {chunk.start.date()} to {chunk.end.date()}")
        data = _normalize_earnings(
            _read_query(
                conn,
                EARNINGS_SQL,
                (
                    chunk.start.strftime("%Y-%m-%d"),
                    chunk.end.strftime("%Y-%m-%d"),
                ),
            )
        )
        if not data.empty:
            _atomic_parquet(data, target)
        summaries.append(
            {
                "dataset": "earnings_pit",
                "start": str(chunk.start.date()),
                "end": str(chunk.end.date()),
                "rows": len(data),
                "status": "written" if not data.empty else "empty",
                "path": str(target),
            }
        )
        print(f"  rows: {len(data):,}")
    return summaries


def export_shares(
    conn,
    output_dir: Path,
    end_date: str | pd.Timestamp,
    resume: bool,
) -> list[dict]:
    target = output_dir / "share_changes" / "share_changes.parquet"
    if resume and target.exists():
        return [
            {
                "dataset": "share_changes",
                "start": None,
                "end": str(pd.Timestamp(end_date).date()),
                "rows": None,
                "status": "skipped",
                "path": str(target),
            }
        ]
    print("share_changes: full PIT history")
    data = _read_query(
        conn,
        SHARES_SQL,
        (pd.Timestamp(end_date).strftime("%Y-%m-%d"),),
    )
    if not data.empty:
        for column in ("PUBLISH_DATE", "CHANGE_DATE"):
            data[column] = pd.to_datetime(data[column], errors="coerce")
        data["SECURITY_ID"] = pd.to_numeric(
            data["SECURITY_ID"], errors="raise"
        ).astype("int64")
        for column in (
            "A_SHARES",
            "B_SHARES",
            "H_SHARES",
            "TOTAL_SHARES",
        ):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.drop_duplicates(
            ["SECURITY_ID", "PUBLISH_DATE", "CHANGE_DATE", "ID"],
            keep="last",
        ).sort_values(
            ["SECURITY_ID", "PUBLISH_DATE", "CHANGE_DATE", "ID"]
        )
        _atomic_parquet(data, target)
    print(f"  rows: {len(data):,}")
    return [
        {
            "dataset": "share_changes",
            "start": None,
            "end": str(pd.Timestamp(end_date).date()),
            "rows": len(data),
            "status": "written" if not data.empty else "empty",
            "path": str(target),
        }
    ]


def download_all(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    warmup_days: int = 400,
    resume: bool = True,
) -> pd.DataFrame:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    requested_start = pd.Timestamp(start_date).normalize()
    market_start = requested_start - timedelta(days=int(warmup_days))
    end = pd.Timestamp(end_date).normalize()

    conn = connect_mysql()
    try:
        summaries = []
        summaries.extend(
            export_market(conn, root, market_start, end, resume=resume)
        )
        summaries.extend(
            export_earnings(
                conn,
                root,
                market_start,
                end,
                resume=resume,
            )
        )
        summaries.extend(export_shares(conn, root, end, resume=resume))
    finally:
        conn.close()

    result = pd.DataFrame(summaries)
    metadata = {
        "requested_start_date": str(requested_start.date()),
        "market_and_pit_start_date": str(market_start.date()),
        "end_date": str(end.date()),
        "warmup_days": int(warmup_days),
        "a_share_type_id": A_SHARE_TYPE_ID,
        "market_value_field": "mkt_equd.MARKET_VALUE",
        "adjusted_close_field": "mkt_equd_adj_af.CLOSE_PRICE_2",
        "earnings_field": "fdmt_main_indi_pit.N_INCOME_CUT",
        "share_count_field": "equ_share_change.TOTAL_SHARES",
    }
    (root / "download_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.to_csv(root / "download_summary.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local Parquet inputs for CH-3/CH-4."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end-date",
        default=str(pd.Timestamp(DEFAULT_END_DATE).date()),
    )
    parser.add_argument("--warmup-days", type=int, default=400)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rewrite partitions that already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = download_all(
        output_dir=arguments.output_dir,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        warmup_days=arguments.warmup_days,
        resume=not arguments.no_resume,
    )
    print(summary.groupby(["dataset", "status"]).size().to_string())

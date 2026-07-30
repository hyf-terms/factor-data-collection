"""Download three 2018-standard A-share PIT statements to local Parquet.

DataYes APIs:
    getFdmtBs2018  - consolidated balance sheet (PIT)
    getFdmtIS2018  - consolidated income statement (PIT)
    getFdmtCF2018  - consolidated cash-flow statement (PIT)

The script reads ``DATAYES_API_TOKEN`` from the adjacent ``.env`` file.
It does not use MySQL, ArcticDB, or any other local database.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_API_BASE_URL = (
    "https://api.wmcloud.com/data/v1/api/fundamental"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "new_pit"


@dataclass(frozen=True)
class PitTable:
    api_name: str
    product_id: int
    source_table: str
    dataset_name: str
    chinese_name: str
    event_key: tuple[str, ...]


PIT_TABLES = (
    PitTable(
        api_name="getFdmtBs2018",
        product_id=2976,
        source_table="vw_fdmt_bs_new",
        dataset_name="new_pit_balance",
        chinese_name="新准则合并资产负债表PIT",
        event_key=(
            "PARTY_ID",
            "END_DATE",
            "ACT_PUBTIME",
            "MERGED_FLAG",
            "END_DATE_REP",
        ),
    ),
    PitTable(
        api_name="getFdmtIS2018",
        product_id=3042,
        source_table="vw_fdmt_is_new",
        dataset_name="new_pit_income",
        chinese_name="新准则合并利润表PIT",
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
        api_name="getFdmtCF2018",
        product_id=2993,
        source_table="vw_fdmt_cf_new",
        dataset_name="new_pit_cashflow",
        chinese_name="新准则合并现金流量表PIT",
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


DATE_COLUMNS = (
    "ACT_PUBTIME",
    "PUBLISH_DATE",
    "END_DATE_REP",
    "END_DATE",
    "UPDATE_TIME",
)

INTEGER_COLUMNS = ("ID", "PARTY_ID", "SECURITY_ID")

TEXT_COLUMNS = (
    "SEC_ID",
    "SEC_SHORT_NAME",
    "TICKER",
    "TICKER_SYMBOL",
    "EXCHANGE_CD",
    "REPORT_TYPE",
    "MERGED_FLAG",
    "ACCOUNTING_STANDARDS",
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


def _date_chunks(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    days: int = 7,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield non-overlapping publication-date intervals."""
    if days < 1:
        raise ValueError("days must be at least 1")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(
            f"start_date {start.date()} is after end_date {end.date()}"
        )

    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _dataset_dir(output_dir: str | Path, spec: PitTable) -> Path:
    return Path(output_dir).expanduser().resolve() / spec.dataset_name


def _partition_path(
    dataset_dir: Path,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
) -> Path:
    return (
        dataset_dir
        / f"year={chunk_start.year}"
        / f"part-{chunk_start:%Y%m%d}-{chunk_end:%Y%m%d}.parquet"
    )


def _upper_snake(name: str) -> str:
    """Convert DataYes camelCase field names to upper snake case."""
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    return second.replace("-", "_").upper()


def _normalize_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.rename(columns={name: _upper_snake(name) for name in df})

    # Names used by the existing factor-building code.
    aliases = {
        "TICKER": "TICKER_SYMBOL",
        "PARTY_ID": "SECURITY_ID",
    }
    for source, target in aliases.items():
        if source in result and target not in result:
            result[target] = result[source]

    # Keep the historical misspelling for compatibility with old notebooks.
    if (
        "ACCOUNTING_STANDARDS" in result
        and "ACCOUTING_STANDARDS" not in result
    ):
        result["ACCOUTING_STANDARDS"] = result["ACCOUNTING_STANDARDS"]

    return result


def _synthetic_ids(df: pd.DataFrame, spec: PitTable) -> pd.Series:
    key_columns = [column for column in spec.event_key if column in df]
    if not key_columns:
        raise KeyError("Cannot create stable IDs without PIT event columns")
    return pd.util.hash_pandas_object(
        df[key_columns].astype("string"),
        index=False,
    ).astype("int64")


def _prepare_chunk(df: pd.DataFrame, spec: PitTable) -> pd.DataFrame:
    if df.empty:
        return df

    result = _normalize_api_columns(df).drop_duplicates().copy()
    required = {
        "PARTY_ID",
        "SECURITY_ID",
        "PUBLISH_DATE",
        "ACT_PUBTIME",
        "END_DATE",
        "END_DATE_REP",
        "FISCAL_PERIOD",
        "MERGED_FLAG",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise KeyError(f"{spec.api_name} is missing required fields: {missing}")

    for column in DATE_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).astype("datetime64[ns]")

    result = result.dropna(
        subset=["PUBLISH_DATE", "ACT_PUBTIME", "END_DATE", "END_DATE_REP"]
    ).copy()

    # API is consolidated+parent-company. Factor work uses consolidated rows.
    result["MERGED_FLAG"] = result["MERGED_FLAG"].astype("string")
    result = result.loc[result["MERGED_FLAG"].eq("1")].copy()
    result = result.loc[
        result["REPORT_TYPE"].astype("string").isin(("A", "Q1", "S1", "Q3"))
    ].copy()

    if "ID" not in result:
        result["ID"] = _synthetic_ids(result, spec)

    duplicate_mask = result.duplicated(list(spec.event_key), keep=False)
    if duplicate_mask.any():
        examples = (
            result.loc[duplicate_mask, list(spec.event_key)]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{spec.chinese_name} has {int(duplicate_mask.sum())} "
            f"duplicate PIT events; examples: {examples}"
        )

    result["IS_CURRENT_PERIOD"] = result["END_DATE"].eq(
        result["END_DATE_REP"]
    )

    for column in INTEGER_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            ).astype("int64")

    result["FISCAL_PERIOD"] = pd.to_numeric(
        result["FISCAL_PERIOD"],
        errors="raise",
    ).astype("int16")

    for column in TEXT_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")

    value_columns = [
        column for column in result if column not in METADATA_COLUMNS
    ]
    if value_columns:
        result[value_columns] = result[value_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).astype("float64")

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
    session,
    token: str,
    spec: PitTable,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_base_url: str = DEFAULT_API_BASE_URL,
    tickers: str | None = None,
    timeout: int = 180,
) -> pd.DataFrame:
    params = {
        "field": "",
        "publishDateBeDate": start_date.strftime("%Y%m%d"),
        "publishDateErDate": end_date.strftime("%Y%m%d"),
        "reportType": "A,Q1,S1,Q3",
    }
    if tickers:
        params["ticker"] = tickers

    response = session.get(
        f"{api_base_url.rstrip('/')}/{spec.api_name}.json",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    ret_code = payload.get("retCode")
    if ret_code == -403:
        raise PermissionError(
            f"DataYes returned Need Privilege for {spec.api_name} "
            f"(product {spec.product_id}). Activate or purchase this product "
            "before running the downloader."
        )
    if ret_code not in (0, 1):
        raise RuntimeError(
            f"DataYes {spec.api_name} failed: "
            f"{ret_code} {payload.get('retMsg', '')}"
        )

    return _prepare_chunk(pd.DataFrame(payload.get("data") or []), spec)


def _write_partition_atomic(
    data: pd.DataFrame,
    target: Path,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Target partition already exists: {target}")

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
    token: str,
    spec: PitTable,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp | None = None,
    resume: bool = True,
    chunk_days: int = 7,
    tickers: str | None = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    session=None,
) -> dict:
    """Download one PIT statement as publication-date Parquet partitions."""
    end = (
        pd.Timestamp.today().normalize()
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )
    start = pd.Timestamp(start_date).normalize()
    dataset_dir = _dataset_dir(output_dir, spec)
    if session is None:
        import requests

        client = requests.Session()
    else:
        client = session
    close_client = session is None

    summary = {
        "api_name": spec.api_name,
        "product_id": spec.product_id,
        "source_table": spec.source_table,
        "dataset": spec.dataset_name,
        "output_dir": str(dataset_dir),
        "rows": 0,
        "chunks": 0,
        "skipped_chunks": 0,
        "start_date": start,
        "end_date": end,
    }

    try:
        for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
            target = _partition_path(dataset_dir, chunk_start, chunk_end)
            if resume and target.exists():
                summary["skipped_chunks"] += 1
                continue

            print(
                f"{spec.chinese_name}: publication dates "
                f"{chunk_start.date()} to {chunk_end.date()}"
            )
            data = _read_chunk(
                client,
                token,
                spec,
                chunk_start,
                chunk_end,
                api_base_url=api_base_url,
                tickers=tickers,
            )
            if data.empty:
                print("  no data")
                continue

            path = _write_partition_atomic(data, target)
            summary["rows"] += len(data)
            summary["chunks"] += 1
            print(f"  wrote {len(data):,} rows to {path}")
    finally:
        if close_client:
            client.close()

    return summary


def download_all_new_pit(
    token: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp | None = None,
    resume: bool = True,
    chunk_days: int = 7,
    tickers: str | None = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> pd.DataFrame:
    """Download all three new-standard PIT statements."""
    import requests

    with requests.Session() as session:
        summaries = [
            download_one_table(
                token=token,
                output_dir=output_dir,
                spec=spec,
                start_date=start_date,
                end_date=end_date,
                resume=resume,
                chunk_days=chunk_days,
                tickers=tickers,
                api_base_url=api_base_url,
                session=session,
            )
            for spec in PIT_TABLES
        ]

    result = pd.DataFrame(summaries)
    print("\nDownload summary")
    print(
        result[
            [
                "api_name",
                "dataset",
                "rows",
                "chunks",
                "skipped_chunks",
                "output_dir",
            ]
        ].to_string(index=False)
    )
    return result


def read_pit_dataset(
    dataset: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one complete local PIT Parquet dataset."""
    path = Path(output_dir).expanduser().resolve() / dataset
    if not path.exists():
        raise FileNotFoundError(f"PIT dataset does not exist: {path}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def _load_environment() -> dict[str, str | None]:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().with_name(".env"))
    token = os.getenv("DATAYES_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing DATAYES_API_TOKEN. Save the API token in the local "
            ".env file; do not commit it to GitHub."
        )
    return {
        "token": token,
        "output_dir": os.getenv(
            "PIT_OUTPUT_DIR",
            str(DEFAULT_OUTPUT_DIR),
        ),
        "start_date": os.getenv("PIT_START_DATE", "2018-01-01"),
        "end_date": os.getenv("PIT_END_DATE") or None,
        "chunk_days": os.getenv("PIT_CHUNK_DAYS", "7"),
        "tickers": os.getenv("PIT_TICKERS") or None,
        "api_base_url": os.getenv(
            "DATAYES_API_BASE_URL",
            DEFAULT_API_BASE_URL,
        ),
    }


if __name__ == "__main__":
    config = _load_environment()
    download_all_new_pit(
        token=str(config["token"]),
        output_dir=str(config["output_dir"]),
        start_date=str(config["start_date"]),
        end_date=config["end_date"],
        chunk_days=int(str(config["chunk_days"])),
        tickers=config["tickers"],
        api_base_url=str(config["api_base_url"]),
        resume=True,
    )

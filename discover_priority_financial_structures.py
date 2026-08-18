"""Discover and profile tables for five under-tested financial structures.

The script queries only metadata plus aggregate row/date/company counts.  It
does not download table contents and imports credentials from the user's local
``download_new_pit_mysql.py`` configuration.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


KEYWORDS = {
    "debt_maturity": [
        "到期", "期限", "偿还", "借款明细", "债务明细", "债券明细",
        "maturity", "deadline", "repay", "borrow", "bond payable",
    ],
    "statement_revision": [
        "修订", "更正", "调整", "差错", "版本", "restatement", "revision",
        "correct", "adjusted flag", "is latest", "actual disclosure time",
    ],
    "audit_text": [
        "审计意见", "关键审计事项", "审计事项", "audit opinion",
        "audit matter", "audit project",
    ],
    "subsidiary": [
        "子公司", "附属公司", "联营", "合营", "subsidiary", "affiliate",
        "joint venture",
    ],
    "contract_execution": [
        "重大合同", "合同负债", "合同资产", "项目进度", "履约", "订单",
        "contract", "performance obligation", "project progress", "order",
    ],
}

IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
DATE_CANDIDATES = [
    "ACT_PUBTIME", "PUBLISH_DATE", "INTL_PUBLISH_DATE", "END_DATE",
    "UPDATE_TIME", "TMSTAMP",
]
ENTITY_CANDIDATES = ["PARTY_ID", "SECURITY_ID", "TICKER_SYMBOL", "SEC_CODE"]


def classify(text: str) -> str:
    folded = text.casefold()
    return ";".join(
        category
        for category, terms in KEYWORDS.items()
        if any(term.casefold() in folded for term in terms)
    )


def quote(name: str) -> str:
    if not IDENTIFIER.fullmatch(name):
        raise ValueError(f"unsafe identifier: {name}")
    return f"`{name}`"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--tables", nargs="*")
    args = parser.parse_args()
    sys.path.insert(0, str(args.config_dir.resolve()))
    from download_new_pit_mysql import connect_mysql

    connection = connect_mysql()
    try:
        columns = pd.read_sql(
            """
            SELECT t.TABLE_NAME, t.TABLE_COMMENT, c.ORDINAL_POSITION,
                   c.COLUMN_NAME, c.COLUMN_COMMENT, c.DATA_TYPE
            FROM information_schema.TABLES t
            JOIN information_schema.COLUMNS c
              ON c.TABLE_SCHEMA=t.TABLE_SCHEMA AND c.TABLE_NAME=t.TABLE_NAME
            WHERE t.TABLE_SCHEMA=DATABASE()
            ORDER BY t.TABLE_NAME,c.ORDINAL_POSITION
            """,
            connection,
        )
        text = columns[["TABLE_NAME", "TABLE_COMMENT", "COLUMN_NAME", "COLUMN_COMMENT"]].fillna("").agg(" ".join, axis=1)
        columns["ROW_CATEGORY"] = text.map(classify)
        table_category = columns.groupby("TABLE_NAME", sort=False)["ROW_CATEGORY"].apply(
            lambda values: ";".join(sorted({x for value in values for x in value.split(";") if x}))
        )
        selected = columns[columns.TABLE_NAME.isin(table_category[table_category.ne("")].index)].copy()
        selected["CATEGORY"] = selected.TABLE_NAME.map(table_category)
        if args.tables:
            selected = selected[selected.TABLE_NAME.isin(args.tables)].copy()

        profiles = []
        for table_name, group in selected.groupby("TABLE_NAME", sort=True):
            names = set(group.COLUMN_NAME.astype(str))
            date_column = next((x for x in DATE_CANDIDATES if x in names), None)
            entity_column = next((x for x in ENTITY_CANDIDATES if x in names), None)
            if args.metadata_only:
                profiles.append({
                    "CATEGORY": table_category[table_name],
                    "TABLE_NAME": table_name,
                    "TABLE_COMMENT": group.TABLE_COMMENT.iloc[0],
                    "DATE_COLUMN": date_column,
                    "ENTITY_COLUMN": entity_column,
                    "COLUMN_COUNT": len(group),
                    "PROFILE_STATUS": "metadata_only",
                })
                continue
            expressions = ["COUNT(*) AS ROW_COUNT"]
            if date_column:
                expressions += [
                    f"MIN({quote(date_column)}) AS MIN_DATE",
                    f"MAX({quote(date_column)}) AS MAX_DATE",
                ]
            if entity_column:
                expressions.append(f"COUNT(DISTINCT {quote(entity_column)}) AS ENTITY_COUNT")
            sql = f"SELECT {', '.join(expressions)} FROM {quote(table_name)}"
            try:
                profile = pd.read_sql(sql, connection).iloc[0].to_dict()
                status = "ok"
            except Exception as exc:  # preserve discovery if a view is restricted
                profile = {}
                status = f"error:{type(exc).__name__}"
            profiles.append({
                "CATEGORY": table_category[table_name],
                "TABLE_NAME": table_name,
                "TABLE_COMMENT": group.TABLE_COMMENT.iloc[0],
                "DATE_COLUMN": date_column,
                "ENTITY_COLUMN": entity_column,
                "COLUMN_COUNT": len(group),
                "PROFILE_STATUS": status,
                **profile,
            })
    finally:
        connection.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "candidate_columns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profiles).to_csv(args.output_dir / "candidate_profiles.csv", index=False, encoding="utf-8-sig")
    print(f"tables={len(profiles)}, columns={len(selected)}")


if __name__ == "__main__":
    main()

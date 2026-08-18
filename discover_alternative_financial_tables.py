"""Discover accessible MySQL tables for alternative financial signals.

Only ``information_schema`` is queried.  Credentials remain in the local
``new_pit_db_local.py`` file imported by ``download_new_pit_mysql.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


CATEGORY_KEYWORDS = {
    "analyst_forecast": [
        "forecast", "consensus", "estimate", "analyst", "research",
        "盈利预测", "一致预期", "分析师", "预测修正", "研报",
    ],
    "performance_preannouncement": [
        "express", "preannounce", "performance", "业绩预告", "业绩快报",
    ],
    "audit": ["audit", "审计意见", "关键审计事项"],
    "accounting_details": [
        "aging", "impair", "inventory", "receiv", "账龄", "减值",
        "存货", "应收",
    ],
    "orders_contract_segment": [
        "order", "contract", "segment", "订单", "合同负债", "分部收入",
        "分部", "主营构成", "主营业务构成", "产品构成", "地区构成", "中标",
    ],
    "ownership_financing": [
        "buyback", "repurchase", "holder", "sharehold", "financing",
        "回购", "增持", "减持", "股东", "融资",
    ],
}


def _categories(text: str) -> str:
    value = text.casefold()
    hits = [
        name
        for name, keywords in CATEGORY_KEYWORDS.items()
        if any(keyword.casefold() in value for keyword in keywords)
    ]
    return ";".join(hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.config_dir.resolve()))
    from download_new_pit_mysql import connect_mysql

    connection = connect_mysql()
    try:
        metadata = pd.read_sql(
            """
            SELECT t.TABLE_NAME, t.TABLE_COMMENT, c.ORDINAL_POSITION,
                   c.COLUMN_NAME, c.COLUMN_COMMENT, c.DATA_TYPE
            FROM information_schema.TABLES AS t
            JOIN information_schema.COLUMNS AS c
              ON c.TABLE_SCHEMA = t.TABLE_SCHEMA
             AND c.TABLE_NAME = t.TABLE_NAME
            WHERE t.TABLE_SCHEMA = DATABASE()
            ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
            """,
            connection,
        )
    finally:
        connection.close()

    combined = metadata[
        ["TABLE_NAME", "TABLE_COMMENT", "COLUMN_NAME", "COLUMN_COMMENT"]
    ].fillna("").agg(" ".join, axis=1)
    metadata["CATEGORY"] = combined.map(_categories)
    table_categories = (
        metadata.groupby("TABLE_NAME", sort=False)["CATEGORY"]
        .apply(lambda values: ";".join(sorted({x for value in values for x in value.split(";") if x})))
    )
    selected_names = table_categories[table_categories.ne("")].index
    result = metadata.loc[metadata["TABLE_NAME"].isin(selected_names)].copy()
    result["CATEGORY"] = result["TABLE_NAME"].map(table_categories)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = (
        result.groupby(["CATEGORY", "TABLE_NAME", "TABLE_COMMENT"], dropna=False)
        .size().reset_index(name="COLUMN_COUNT")
    )
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"candidate tables: {len(summary)}")
    print(f"columns: {args.output.resolve()}")
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()

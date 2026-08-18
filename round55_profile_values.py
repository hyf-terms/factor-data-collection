"""Small read-only value audit for round 55 source tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


QUERIES = {
    "inventory_schema": """
        SELECT t.TABLE_NAME, t.TABLE_COMMENT, c.COLUMN_NAME, c.COLUMN_COMMENT
        FROM information_schema.TABLES t
        JOIN information_schema.COLUMNS c
          ON c.TABLE_SCHEMA=t.TABLE_SCHEMA AND c.TABLE_NAME=t.TABLE_NAME
        WHERE t.TABLE_SCHEMA=DATABASE()
          AND (LOWER(t.TABLE_NAME) LIKE '%inventory%' OR t.TABLE_COMMENT LIKE '%存货%'
               OR c.COLUMN_COMMENT LIKE '%原材料%' OR c.COLUMN_COMMENT LIKE '%产成品%'
               OR c.COLUMN_COMMENT LIKE '%库存商品%')
        ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
    """,
    "concentration_schema": """
        SELECT t.TABLE_NAME, t.TABLE_COMMENT, c.COLUMN_NAME, c.COLUMN_COMMENT
        FROM information_schema.TABLES t
        JOIN information_schema.COLUMNS c
          ON c.TABLE_SCHEMA=t.TABLE_SCHEMA AND c.TABLE_NAME=t.TABLE_NAME
        WHERE t.TABLE_SCHEMA=DATABASE()
          AND (t.TABLE_COMMENT LIKE '%客户%' OR t.TABLE_COMMENT LIKE '%供应商%'
               OR t.TABLE_COMMENT LIKE '%采购%' OR t.TABLE_COMMENT LIKE '%主营业务构成%'
               OR c.COLUMN_COMMENT LIKE '%前五大客户%' OR c.COLUMN_COMMENT LIKE '%前五大供应商%'
               OR c.COLUMN_COMMENT LIKE '%客户集中%' OR c.COLUMN_COMMENT LIKE '%供应商集中%')
        ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
    """,
    "related_type": """
        SELECT TYPE, ITEM_NAME, COUNT(*) AS N,
               COUNT(DISTINCT PARTY_ID) AS FIRMS,
               MIN(ACT_PUBTIME) AS MIN_TIME, MAX(ACT_PUBTIME) AS MAX_TIME
        FROM fdmt_related_rec_pay
        WHERE ACT_PUBTIME >= '2016-01-01'
        GROUP BY TYPE, ITEM_NAME ORDER BY N DESC LIMIT 100
    """,
    "inventory_item": """
        SELECT ITEM_ID, COUNT(*) AS N, COUNT(DISTINCT PARTY_ID) AS FIRMS,
               MIN(ACT_PUBTIME) AS MIN_TIME, MAX(ACT_PUBTIME) AS MAX_TIME
        FROM fdmt_bs_inventory
        WHERE ACT_PUBTIME >= '2016-01-01'
        GROUP BY ITEM_ID ORDER BY N DESC LIMIT 100
    """,
    "inventory_item_dictionary": """
        SELECT ITEM_ID, ITEM_NAME FROM fdmt_mo_item
        WHERE ITEM_ID BETWEEN 101600 AND 101900 ORDER BY ITEM_ID
    """,
    "trader_type": """
        SELECT TRADER_TYPE_CD, COUNT(*) AS N, COUNT(DISTINCT PARTY_ID) AS FIRMS,
               MIN(PUBLISH_DATE) AS MIN_DATE, MAX(PUBLISH_DATE) AS MAX_DATE,
               MIN(TRADER_RANK) AS MIN_RANK, MAX(TRADER_RANK) AS MAX_RANK
        FROM fdmt_trader
        WHERE PUBLISH_DATE >= '2016-01-01'
        GROUP BY TRADER_TYPE_CD ORDER BY N DESC
    """,
    "inventory_flags": """
        SELECT REPORT_TYPE, MERGED_FLAG, IS_NEW, ADJUSTED_FLAG,
               COUNT(*) AS N, COUNT(DISTINCT PARTY_ID) AS FIRMS
        FROM fdmt_bs_inventory
        WHERE ACT_PUBTIME >= '2016-01-01'
        GROUP BY REPORT_TYPE, MERGED_FLAG, IS_NEW, ADJUSTED_FLAG
        ORDER BY N DESC LIMIT 50
    """,
    "related_flags": """
        SELECT REPORT_TYPE, MERGED_FLAG, IS_NEW, ADJUSTED_FLAG,
               COUNT(*) AS N, COUNT(DISTINCT PARTY_ID) AS FIRMS
        FROM fdmt_related_rec_pay
        WHERE ACT_PUBTIME >= '2016-01-01'
        GROUP BY REPORT_TYPE, MERGED_FLAG, IS_NEW, ADJUSTED_FLAG
        ORDER BY N DESC LIMIT 50
    """,
    "top5_flags": """
        SELECT REPORT_TYPE, MERGED_FLAG, IS_NEW,
               COUNT(*) AS N, COUNT(DISTINCT PARTY_ID) AS FIRMS,
               MIN(ACT_PUBTIME) AS MIN_TIME, MAX(ACT_PUBTIME) AS MAX_TIME
        FROM fdmt_rec_top5
        WHERE ACT_PUBTIME >= '2016-01-01'
        GROUP BY REPORT_TYPE, MERGED_FLAG, IS_NEW ORDER BY N DESC LIMIT 50
    """,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.config_dir.resolve()))
    from download_new_pit_mysql import connect_mysql

    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = connect_mysql()
    try:
        for name, sql in QUERIES.items():
            frame = pd.read_sql(sql, connection)
            frame.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            print(f"{name}: {len(frame)} rows")
            print(frame.head(20).to_string(index=False))
    finally:
        connection.close()


if __name__ == "__main__":
    main()

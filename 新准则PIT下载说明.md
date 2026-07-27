# 新准则三大报表 PIT 下载说明

## 数据表

程序已根据数据库字段核对使用以下三张表：

| 报表 | 数据源 | ArcticDB 目标 symbol |
|---|---|---|
| 资产负债表 | `vw_fdmt_bs_new` | `new_pit_balance` |
| 利润表 | `vw_fdmt_is_new` | `new_pit_income` |
| 现金流量表 | `vw_fdmt_cf_new` | `new_pit_cashflow` |

目标 symbol 使用新名称，不会覆盖已有的 `balance`、`income`、`cashflow`。

## 推荐运行方法

先在 `history_data.ipynb` 中运行原有的数据库连接和 ArcticDB 连接代码，确保
`conn`、`lib` 已存在，然后新建一个单元格：

```python
from download_new_pit import download_all_new_pit

summary = download_all_new_pit(
    conn=conn,
    lib=lib,
    start_date="2018-01-01",
    end_date="2026-07-27",
    resume=True,
)
summary
```

程序按自然年分段读取，成功写入一个年度后才继续下一年度。重新运行时会读取
ArcticDB 中已有的最大 `PUBLISH_DATE`，从下一日继续，避免重复追加。

## 数据口径

- 只保留合并报表：`MERGED_FLAG='1'`。
- 只保留年报、一季报、半年报、三季报：`A/Q1/S1/Q3`。
- 使用证券在公告时点有效的上市及证券类型关系。
- 保留当前期、比较期和后续修订记录。
- 保留 `PUBLISH_DATE`、`ACT_PUBTIME`、`UPDATE_TIME` 等 PIT 字段。
- 以 `PUBLISH_DATE` 作为 ArcticDB 时间索引。
- 添加 `IS_CURRENT_PERIOD`，用于区分当前期和比较期。

## PIT 使用注意

下载完成的数据仍是“公告事件表”，不能直接按照财报截止日合并到每日行情。
生成因子时应根据 `ACT_PUBTIME` 确定信息可用日：

- 开盘前公告：可根据策略约定在当日使用。
- 交易时段或收盘后公告：建议从下一交易日使用。
- 每个股票、交易日只选择当时已经公开的最新版本。

不要使用 `END_DATE` 作为信息可用日期，也不要只保留数据库当前的最终版本。

## 命令行运行

如需直接运行 `download_new_pit.py`，请预先设置：

```text
PIT_DB_HOST
PIT_DB_PORT
PIT_DB_USER
PIT_DB_PASSWORD
PIT_DB_NAME
PIT_ARCTIC_URI
PIT_ARCTIC_LIBRARY
PIT_START_DATE
PIT_END_DATE
```

数据库密码没有写入程序文件。

## 增量更新限制

断点续跑按照 `PUBLISH_DATE` 继续，适合首次历史下载和正常向后更新。如果数据供应商
使用 `UPDATE_TIME` 修正了较早公告日的原记录，单纯向后更新不会捕获这种回填；建议定期
用新 symbol 做一次完整重建，再核对事件数量。

# 新准则三大报表 PIT 本地下载说明

程序通过 DataYes Data API 下载数据，并直接写成本地分区 Parquet：

```text
DataYes API（只读）
        -> 按实际披露日期分段请求
        -> 字段标准化
        -> 本地 Parquet
```

不需要 MySQL、ArcticDB 或其他本地数据库。

## 三张表

| 报表 | API | 产品 ID | 本地数据集 |
|---|---|---:|---|
| 新准则合并资产负债表 PIT | `getFdmtBs2018` | 2976 | `new_pit_balance` |
| 新准则合并利润表 PIT | `getFdmtIS2018` | 3042 | `new_pit_income` |
| 新准则合并现金流量表 PIT | `getFdmtCF2018` | 2993 | `new_pit_cashflow` |

这三个 API 必须在 DataYes 账号中已开通。未开通时接口返回
`-403 Need Privilege`，程序会停止，不会自动申请试用或付费。

## 本地配置

`.env` 已被 `.gitignore` 排除，不会同步到 GitHub。配置示例：

```text
DATAYES_API_TOKEN=你的Token
DATAYES_API_BASE_URL=https://api.wmcloud.com/data/v1/api/fundamental
PIT_OUTPUT_DIR=C:/Users/hyf/Desktop/因子/data/new_pit
PIT_START_DATE=2018-01-01
PIT_END_DATE=2026-07-28
PIT_CHUNK_DAYS=7
PIT_TICKERS=
```

`PIT_TICKERS` 留空表示请求全部股票；也可填写逗号分隔的股票代码用于小范围
测试。默认按 7 天披露区间请求，便于控制单次响应体积，并支持断点续传。

## 运行

```powershell
cd "C:\Users\hyf\Desktop\因子"
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" .\download_new_pit.py
```

输出结构：

```text
C:\Users\hyf\Desktop\因子\data\new_pit\
├── new_pit_balance\year=2018\part-20180101-20180107.parquet
├── new_pit_income\year=2018\part-20180101-20180107.parquet
└── new_pit_cashflow\year=2018\part-20180101-20180107.parquet
```

再次运行时，已有日期分段会被跳过。

## 字段处理

- API 的 camelCase 字段统一转换为大写下划线形式。
- `PARTY_ID` 同时写入 `SECURITY_ID`，与现有标签和因子程序衔接。
- 只保留 `MERGED_FLAG == "1"` 的合并报表。
- 只保留 `A/Q1/S1/Q3` 四类定期报告。
- 保留 `PUBLISH_DATE`、`ACT_PUBTIME`、`END_DATE_REP`、`END_DATE` 和
  `UPDATE_TIME`。
- 添加 `IS_CURRENT_PERIOD`。
- API 未提供数据库自增 `ID` 时，按 PIT 事件键生成稳定的本地 `ID`。

因子构建必须用 `ACT_PUBTIME` 判断数据最早可用时点，不能按报告期
`END_DATE` 提前回填。

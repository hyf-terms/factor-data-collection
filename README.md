# Factor Data Collection

用于收集因子研究所需的基础数据。当前版本先保存新会计准则下三大财务报表的原始披露 PIT 数据下载流程。

## 当前内容

- `download_new_pit.py`：按年下载资产负债表、利润表和现金流量表 PIT 数据。
- `新准则PIT下载说明.md`：Notebook 与命令行使用说明。
- `new_pit.env.example`：不含真实账号密码的环境变量模板。

## 数据源

| 报表 | 数据源表 | ArcticDB symbol |
|---|---|---|
| 资产负债表 | `vw_fdmt_bs_new` | `new_pit_balance` |
| 利润表 | `vw_fdmt_is_new` | `new_pit_income` |
| 现金流量表 | `vw_fdmt_cf_new` | `new_pit_cashflow` |

程序保留 `PUBLISH_DATE`、`ACT_PUBTIME` 和修订记录，用于后续按历史可得信息构造 PIT 因子。

## 安全说明

仓库不保存数据库密码、Parquet 数据、ArcticDB 数据库或其他大体积本地数据。请复制环境变量模板并在本地配置真实连接信息。

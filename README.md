# Factor Data Collection

用于收集因子研究所需的基础数据。当前版本先保存新会计准则下三大财务报表的原始披露 PIT 数据下载流程。

## 当前内容

- `download_new_pit.py`：按公告年份下载三张PIT表并保存为分区Parquet。
- `新准则PIT下载说明.md`：Notebook 与命令行使用说明。
- `new_pit.env.example`：不含真实账号密码的环境变量模板。
- `requirements.txt`：Parquet读写和数据库连接依赖。
- `gross_profitability_factor.py`：Novy-Marx 毛利盈利因子的严格 PIT 实现。
- `毛利盈利因子复现说明.md`：论文口径、字段映射与运行示例。

## 数据源

| 报表 | 数据源表 | Parquet数据集 |
|---|---|---|
| 资产负债表 | `vw_fdmt_bs_new` | `new_pit_balance` |
| 利润表 | `vw_fdmt_is_new` | `new_pit_income` |
| 现金流量表 | `vw_fdmt_cf_new` | `new_pit_cashflow` |

程序不依赖ArcticDB，默认保存到 `data/new_pit`。它保留 `PUBLISH_DATE`、
`ACT_PUBTIME` 和修订记录，用于后续按历史可得信息构造PIT因子。

## 安全说明

仓库不保存数据库密码、Parquet数据或其他大体积本地数据。请复制环境变量模板并在本地配置真实连接信息。

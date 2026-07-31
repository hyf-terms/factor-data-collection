# Factor Data Collection

用于下载 A 股新准则财务 PIT 数据、构建财务因子，并在 Barra 中性化后检验每日 Spearman IC。

## 主要模块

- `download_new_pit.py`：通过数据接口下载三张 PIT 表并保存为分区 Parquet。
- `download_new_pit_mysql.py`：从 MySQL 数据库读取 PIT 表并保存到本地。
- `download_quarterly_financial_indicators.py`：下载新准则单季度财务指标 PIT，并输出完整字段字典。
- `gross_profitability_factor.py`：毛利盈利因子。
- `pead_sue_factor.py`：PEAD / SUE 盈余惊喜因子。
- `quarterly_f_score.py`：季度 Piotroski F-score。
- `mohanram_g_score.py`：Mohanram G-score。
- `fundamental_priority_factors.py`：经营利润增长、经营利润加速度、CFO SUE、应计质量。
- `fundamental_priority_factors_part2.py`：资产增长、投资率、应收与存货异常增长。
- `secondary_priority_factors.py`：盈利质量、管理层误定价、应计及非经常损益、基本面动量复合因子。
- `event_financial_factor_search.py`：构建 Q1 盈余惊喜、扣非盈余惊喜及事件条件财务复合候选。
- `literature_financial_factor_search.py`：复现收入/利润惊喜、联合惊喜、惊喜持续性并生成受约束的复合参数变体。
- `unreplicated_financial_factor_search.py`：复现净经营资产、杜邦分解、研发强度、资本开支和盈利稳定性等新增文献候选。
- `quarterly_indicator_factor_search.py`：利用单季度财务指标PIT挖掘现金质量、增长确认、回款和偿债候选。
- `ch_factor_models.py`：中国市场 CH-3、CH-4 模型复现。
- `factors_neus_only.py`：合并因子、Barra 和标签，逐日残差化并计算 IC。
- `organize_factor_packages.py`：按最新中性化 IC 将因子、代码、说明和轻量测试结果整理到“有效因子/无效因子”目录。

## 数据口径

| 报表 | 数据源表示例 | 本地数据集 |
|---|---|---|
| 资产负债表 | `vw_fdmt_bs_new` | `new_pit_balance` |
| 利润表 | `vw_fdmt_is_new` | `new_pit_income` |
| 现金流量表 | `vw_fdmt_cf_new` | `new_pit_cashflow` |

构造程序保留披露时间和修订记录，以首次可用时间形成严格 PIT 数据。默认数据、因子组件和测试结果均保存在本地，不进入 Git。

## 快速测试

```powershell
python -m pytest
```

每个模块的字段映射、计算口径和命令行示例见对应中文说明文档。

## 安全说明

仓库不保存真实账号、数据库密码、Parquet 数据或日志。请复制 `new_pit.env.example` 或 `new_pit_db_local.example.py`，在被 Git 忽略的本地文件中填写连接信息。

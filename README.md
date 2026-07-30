# Factor Data Collection

用于下载 A 股新准则财务 PIT 数据、构建财务因子，并在 Barra 中性化后检验每日 Spearman IC。

## 主要模块

- `download_new_pit.py`：通过数据接口下载三张 PIT 表并保存为分区 Parquet。
- `download_new_pit_mysql.py`：从 MySQL 数据库读取 PIT 表并保存到本地。
- `gross_profitability_factor.py`：毛利盈利因子。
- `pead_sue_factor.py`：PEAD / SUE 盈余惊喜因子。
- `quarterly_f_score.py`：季度 Piotroski F-score。
- `mohanram_g_score.py`：Mohanram G-score。
- `fundamental_priority_factors.py`：经营利润增长、经营利润加速度、CFO SUE、应计质量。
- `fundamental_priority_factors_part2.py`：资产增长、投资率、应收与存货异常增长。
- `secondary_priority_factors.py`：盈利质量、管理层误定价、应计及非经常损益、基本面动量复合因子。
- `ch_factor_models.py`：中国市场 CH-3、CH-4 模型复现。
- `factors_neus_only.py`：合并因子、Barra 和标签，逐日残差化并计算 IC。

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

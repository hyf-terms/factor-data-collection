# A股财务因子研究与严格测试

本项目从通联新准则PIT财务数据构造A股财务因子，使用逐日Barra截面回归残差与未来10个交易日标签计算Spearman IC，并从达标因子中筛选低相关组合。

## 当前结论

- 正式测试程序为 `factors_neus_only2.py`；旧版 `factors_neus_only.py` 仅用于复现历史稀疏结果。
- 正式稠密因子必须覆盖共同股票池的每个交易日，不能跳过缺失、恒定或低方差横截面。
- `|平均IC| >= 0.035` 为有效，`0.030 <= |平均IC| < 0.035` 为潜在有效。
- 当前严格达标代表因子共7个，最高为 `r43_profitability_persistence_confirmation_equal6`，严格IC为0.03724。
- 达标因子的每日IC10序列绝对相关性为0.8450—0.9995。按IC降序、相关性严格低于0.85的贪心规则，最终只保留第43轮因子。
- 后续测试的债务、税务、审计、披露、合同、子公司、现金流风险等独立方向均未新增IC达到0.03的稠密因子。

## 核心文档

1. [研究方法与测试规范](研究方法与测试规范.md)：PIT、稠密化、Barra中性化、IC与防过拟合规则。
2. [有效因子与最终结果](有效因子与最终结果.md)：达标因子、潜在因子、相关性和最终选择。
3. [运行与数据获取](运行与数据获取.md)：数据下载、因子构造、正式测试及贪心筛选命令。

历史逐轮Markdown已合并到上述文档。各轮代码仍保留在仓库根目录，便于复现；本地Parquet、测试产物、数据库配置和日志均由 `.gitignore` 排除。

## 关键程序

| 功能 | 程序 |
|---|---|
| 三张新准则PIT下载 | `download_new_pit.py`、`download_new_pit_mysql.py` |
| 通用MySQL表导出Parquet | `download_mysql_table_parquet.py` |
| 正式Barra中性化与IC | `factors_neus_only2.py` |
| 稠密候选填充检查 | `prepare_strict_neutral_fill.py`、`prepare_round65_69_strict_dense.py` |
| 达标因子公共池 | `build_existing_effective_factor_pool.py` |
| 每日IC贪心筛选 | `greedy_ic_factor_selector.py` |
| 因子残差横截面相关性 | `factor_correlation_check.py` |

## 快速校验

```powershell
python -m pytest -q
```

## 安全约束

仓库不保存账号、密码、原始Parquet或日志。请从 `new_pit.env.example` 或 `new_pit_db_local.example.py` 创建仅本地使用的配置文件。

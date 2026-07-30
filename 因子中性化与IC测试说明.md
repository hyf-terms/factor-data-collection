# 因子中性化与 IC 测试

`factors_neus_only.py` 已形成完整测试管线：

1. 读取 `factors.parquet`、`barra_diy.parquet`、`label.parquet`；
2. 按 `TRADE_DATE`、`SECURITY_ID` 合并；
3. 每日使用截距、Barra 风格因子和行业哑变量做 OLS；
4. 以回归残差作为中性化因子；
5. 计算每日截面 Spearman IC；
6. 汇总 IC 均值、绝对值、标准差、ICIR、t 值、胜率和覆盖数。

直接运行：

```powershell
& ".\.venv\Scripts\python.exe" ".\factors_neus_only.py"
```

默认输出目录为 `factor_test_output`：

- `factors_neutralized.parquet`：Barra 残差因子；
- `daily_ic.parquet`：每个交易日和因子的 IC；
- `ic_summary.parquet`、`ic_summary.csv`：汇总指标；
- `merge_diagnostics.parquet`：逐年合并覆盖诊断；
- `run_metadata.json`：参数、日期和 Barra 暴露列。

IC 输出包含三个版本：

- `raw`：原始因子在全部 factor-label 交集上的 IC；
- `raw_matched`：原始因子在可完成 Barra 中性化的同一批样本上的 IC；
- `neutral`：Barra 残差因子的 IC。

比较 `raw_matched` 与 `neutral` 可以避免因股票样本变化造成误判。

指定日期或标签：

```powershell
& ".\.venv\Scripts\python.exe" ".\factors_neus_only.py" `
  --start-date 2020-01-01 `
  --end-date 2025-12-31 `
  --label-column LABEL
```


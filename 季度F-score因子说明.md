# 季度 F-score 因子

## 定义

本项目将 Piotroski（2000）的年度 F-score 改写为季度频率。原始九项信号仍然保留，但利润表和现金流量表先还原成单季度值，变化类指标改为与上年同季度比较，以减少季节性。

九项信号各取 0 或 1 分：

1. 单季度 ROA 大于 0；
2. 单季度经营现金流/期初资产大于 0；
3. ROA 同比改善；
4. 经营现金流/期初资产大于 ROA；
5. 长期债务率同比下降；
6. 流动比率同比上升；
7. 实收资本同比未增加；
8. 毛利率同比上升；
9. 资产周转率同比上升。

总分 `quarterly_f_score` 介于 0 和 9。九项数据必须全部有效才生成分数。

## 字段口径

- ROA：`N_INCOME_ATTR_P / 上季末T_ASSETS`
- CFO：`N_CF_OPERATE_A / 上季末T_ASSETS`
- 长期债务率：`(LT_BORR + BOND_PAYABLE) / T_ASSETS`
- 流动比率：`T_CA / T_CL`
- 未发行权益：本季 `PAID_IN_CAPITAL` 不高于上年同季度
- 毛利率：`(REVENUE - COGS) / REVENUE`
- 资产周转率：`REVENUE / 上季末T_ASSETS`

银行、证券、保险公司的资产负债结构与一般工商企业不可比，因此明确排除。

## PIT 与生效时点

程序只使用合并报表当前期记录，并以 `ACT_PUBTIME` 判断信息何时公开。一个季度分数只有在当前季度、上季末资产和上年同季度比较数据都已经公开后才生效。盘后披露顺延到下一交易日，旧季度的迟到记录不能覆盖已经公开的新季度分数。

## 运行

```powershell
& "C:\Users\hyf\Desktop\因子\.venv\Scripts\python.exe" `
  "C:\Users\hyf\Desktop\因子\quarterly_f_score.py"
```

程序会把 `quarterly_f_score` 作为新列加入 `factors.parquet`，首次运行前备份为 `factors_before_quarterly_f_score.parquet`。

审计文件保存在 `factor_components`：

- `quarterly_f_score_events.parquet`
- `quarterly_f_score_metrics.parquet`
- `quarterly_f_score_daily.parquet`
- `quarterly_f_score_diagnostics.json`

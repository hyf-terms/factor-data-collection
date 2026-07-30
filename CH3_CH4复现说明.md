# CH-3 / CH-4 模型复现说明

论文：Liu, Stambaugh and Yuan, *Size and Value in China*。

本项目将模型因子保存为独立的月度时间序列，不写入股票截面的
`factors.parquet`。`factors.parquet` 用于逐股票、逐日 IC 测试，而 CH-3/CH-4
是用于时间序列回归的可交易组合收益，两者的数据结构和用途不同。

## 一、数据口径

- 股票池：A股。
- 行情：`mkt_equd` 与 `mkt_equd_adj_af`。
- 规模与权重：上月末 A 股总市值 `mkt_equd.MARKET_VALUE`。
- 收益：后复权收盘价 `CLOSE_PRICE_2` 计算的月收益。
- EP 分子：PIT 表 `fdmt_main_indi_pit.N_INCOME_CUT`，即最新已经披露的
  扣除非经常性损益净利润。
- EP 分母：上月末收盘价乘以当时已披露且已生效的全股本，优先使用
  `equ_share_change.TOTAL_SHARES`，其中包含 A/B/H 股。
- 日换手率：成交股数除以 A 股总股本。A 股总股本由
  `MARKET_VALUE / CLOSE_PRICE` 反推，避免使用数据库中主要基于流通股本的
  `TURNOVER_RATE`。
- 无风险利率：默认年化 1.5% 的一年期存款基准利率，换算为月度有效利率。

财务披露若发生在开盘前，可以在当日使用；否则从下一个交易日开始可用。
月末形成组合，持有下一个自然月，避免使用未来收益。

## 二、论文筛选

每个月末：

1. 排除上市不足 183 天的股票。
2. 排除过去 366 天交易记录不足 120 条的股票。
3. 排除过去 31 天交易记录不足 15 条的股票。
4. 按 A 股总市值剔除最小 30%。
5. 对剩余 70% 股票进行独立的规模与风格排序。

## 三、CH-3

剩余股票按规模中位数分为 S/B，按 EP 的 30%、70%分位点分为
G/M/V，负 EP 强制归入 G。形成 SV、SM、SG、BV、BM、BG 六个市值加权
组合：

```text
SMB_EP = mean(SV, SM, SG) - mean(BV, BM, BG)
VMG    = mean(SV, BV) - mean(SG, BG)
MKT    = top-70%股票市值加权收益 - 月度无风险收益
```

## 四、CH-4

异常换手率为过去 31 天平均日换手率除以过去 366 天平均日换手率。按其
30%、70%分位点分为 P/N/O，再与 S/B 交叉形成六个组合：

```text
PMO          = mean(SP, BP) - mean(SO, BO)
SMB_TURNOVER = mean(SP, SN, SO) - mean(BP, BN, BO)
SMB_CH4      = mean(SMB_EP, SMB_TURNOVER)
```

CH-4 使用 `MKT, SMB_CH4, VMG, PMO`。

## 五、运行

下载 2017 年至今模型所需数据。程序会自动向前多下载 400 天，用于构造
首期过去一年换手率：

```powershell
& ".\.venv\Scripts\python.exe" ".\download_ch_model_data.py" `
  --start-date 2017-01-01 `
  --end-date 2026-07-29
```

构造模型：

```powershell
& ".\.venv\Scripts\python.exe" ".\ch_factor_models.py" `
  --start-date 2017-01-01 `
  --end-date 2026-07-29
```

PowerShell 不允许执行 `Activate.ps1` 时无需激活虚拟环境，直接调用
`.venv\Scripts\python.exe` 即可。

## 六、输出

默认输出到 `outputs/ch_models/`：

- `ch3_factors.parquet/csv`：`TRADE_DATE, MKT, SMB, VMG`
- `ch4_factors.parquet/csv`：`TRADE_DATE, MKT, SMB, VMG, PMO`
- `portfolio_returns.parquet`：十二个基础组合及样本数
- `formation_assignments.parquet`：逐股票月末分组、EP、异常换手率和权重
- `monthly_diagnostics.parquet`：每月筛选数量及每个组合样本数
- `factor_summary.csv`：分别列示CH-3和CH-4的均值、波动率、均值 t
  统计量和简单年化均值
- `ch3_factor_correlations.csv`、`ch4_factor_correlations.csv`：两个模型的
  因子相关系数
- `run_metadata.json`：本次运行口径

若某月任一基础组合不足 50 只股票，则对应因子留空，不静默放宽论文约束。
若 `--end-date` 位于尚未结束的月份，程序自动只输出到上一个完整月，避免把
月中累计收益误当成完整月收益。

## 七、与论文结果的比较

论文使用 2000-2016 年样本，而本项目现有标签及因子股票池从 2017 年开始，
因此收益均值不会与论文表 3 完全一致。复现应检查：

- 六个基础组合每月均达到 50 只；
- 所有因子形成月和收益月严格错开一个月；
- EP 财务可用日不晚于组合形成日；
- `reported_total_shares` 的覆盖率；
- CH-3 与 CH-4 因子的月均收益、波动率和相关系数是否经济上合理。

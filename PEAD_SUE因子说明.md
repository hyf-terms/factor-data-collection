# PEAD / SUE 盈余惊喜因子

## 数据

因子使用新准则合并利润表 PIT 的以下字段：

- `SECURITY_ID`
- `ACT_PUBTIME`
- `END_DATE`
- `REPORT_TYPE`
- `FISCAL_PERIOD`
- `N_INCOME_ATTR_P`

为使2017年初具备八季度历史窗口，利润表 PIT 需至少覆盖到2014年。

## 单季度盈余

使用归母净利润而不是累计 EPS，避免累计 EPS 因加权股本变化而无法直接相减：

```text
Q1 = 一季报归母净利润
Q2 = 半年报累计归母净利润 - Q1
Q3 = 三季报披露的单季度归母净利润
     （缺失时使用三季报累计值 - 半年报累计值）
Q4 = 年报累计归母净利润 - 三季报累计值
```

## SUE

```text
UE(t)  = QuarterlyEarnings(t) - QuarterlyEarnings(t-4)
SUE(t) = UE(t) / Std[UE(t-8), ..., UE(t-1)]
```

严格要求前八个季度连续、均已在当时披露，标准差必须大于零。

## PIT规则

1. 每个季度只采用最早的有效财报披露；
2. 使用 `ACT_PUBTIME` 判断信息可得时间；
3. 开盘后披露的信息从下一交易日开始使用；
4. 后续发布的历史重述不能进入更早时点的SUE；
5. 因子按日携带至下一次有效季度盈余惊喜；
6. 每日截面进行1%和99%缩尾。

## 运行

```powershell
& ".\.venv\Scripts\python.exe" ".\pead_sue_factor.py"
```

程序将 `pead_sue` 作为新列加入 `factors.parquet`。首次运行会保留备份：

```text
factors_before_pead_sue.parquet
```

审计文件位于 `factor_components`：

- `pead_sue_events.parquet`
- `pead_sue_daily.parquet`
- `pead_sue_diagnostics.json`


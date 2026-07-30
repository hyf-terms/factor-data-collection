# MySQL 直连 PIT 下载

当 DataYes API 账号没有三张 PIT 产品权限时，可直接从内部 MySQL 读取：

- `vw_fdmt_bs_new`：新准则资产负债表 PIT
- `vw_fdmt_is_new`：新准则利润表 PIT
- `vw_fdmt_cf_new`：新准则现金流量表 PIT

运行：

```powershell
& .\.venv\Scripts\python.exe .\download_new_pit_mysql.py
```

程序默认读取 2016-01-01 至当前日期，按 31 天分块，并保存到：

```text
data/new_pit/new_pit_balance
data/new_pit/new_pit_income
data/new_pit/new_pit_cashflow
```

每个分块是压缩 Parquet 文件。再次运行时会跳过已存在的分块，可断点续传。

数据库连接信息位于 `new_pit_db_local.py`。该文件已由 `.gitignore`
排除，不能提交至 GitHub；仓库只保存无真实密码的
`new_pit_db_local.example.py`。

查询会：

1. 只保留合并报表；
2. 只保留 A、Q1、S1、Q3 报告类型；
3. 通过 `md_security` 和 `md_sec_type` 筛选 A 股并补充
   `SECURITY_ID`；
4. 保留发布日期、实际披露时间、比较期与后续修订记录；
5. 生成与现有因子程序兼容的字段和 Parquet 目录。

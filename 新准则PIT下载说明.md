# 新准则三大报表PIT下载说明

## 与原Notebook的存储关系

`history_data.ipynb` 使用：

```python
ac = adb.Arctic("lmdb://C:/nz/arcticdb?map_size=600GB")
lib = ac["hermes"]
```

并通过 `lib.write()`、`lib.append()` 保存行情和财务数据。为避免新PIT影响
`hermes` 中的历史行情和旧财务表，程序在同一个ArcticDB实例中新建独立库
`factor_pit`：

```text
C:\nz\arcticdb
├── hermes       # 原Notebook数据
└── factor_pit   # 新PIT数据
```

新库包含：

| 报表 | 数据源 | 新ArcticDB symbol |
|---|---|---|
| 资产负债表 | `vw_fdmt_bs_new` | `new_pit_balance` |
| 利润表 | `vw_fdmt_is_new` | `new_pit_income` |
| 现金流量表 | `vw_fdmt_cf_new` | `new_pit_cashflow` |

## 推荐：在Notebook创建并写入新库

先运行Notebook中创建源数据库连接 `conn` 的单元格，再运行：

```python
from download_new_pit import (
    download_all_new_pit,
    open_or_create_arctic_library,
)

pit_lib = open_or_create_arctic_library(
    uri="lmdb://C:/nz/arcticdb?map_size=600GB",
    library_name="factor_pit",
)

summary = download_all_new_pit(
    conn=conn,
    lib=pit_lib,
    storage="arcticdb",
    start_date="2018-01-01",
    end_date="2026-07-28",
    resume=True,
)
summary
```

`open_or_create_arctic_library` 会先检查库名；不存在才创建，不会删除或重建
`hermes`。`resume=True` 会读取每个新symbol的最大 `PUBLISH_DATE`，从下一日
继续。首次使用 `write`，以后按公告年份使用 `append`。

检查结果：

```python
[name for name in pit_lib.list_symbols() if name.startswith("new_pit_")]

for name in ["new_pit_balance", "new_pit_income", "new_pit_cashflow"]:
    tail = pit_lib.tail(name, n=3).data
    print(name, tail.shape, tail.index.min(), tail.index.max())
```

## 可选：保存为分区Parquet

若希望生成便于交换或备份的文件，不传 `lib`：

```python
summary = download_all_new_pit(
    conn=conn,
    output_dir=r"C:\Users\hyf\Desktop\因子\data\new_pit",
    storage="parquet",
    start_date="2018-01-01",
    end_date="2026-07-28",
    resume=True,
)
```

输出结构：

```text
data/new_pit/
├── new_pit_balance/year=YYYY/*.parquet
├── new_pit_income/year=YYYY/*.parquet
└── new_pit_cashflow/year=YYYY/*.parquet
```

## PowerShell命令行模式

在原ArcticDB实例中新建 `factor_pit` 库：

```powershell
cd "C:\Users\hyf\Desktop\因子"

$env:PIT_DB_HOST = "数据库服务器地址"
$env:PIT_DB_PORT = "3306"
$env:PIT_DB_USER = "数据库用户名"
$env:PIT_DB_PASSWORD = "数据库密码"
$env:PIT_DB_NAME = "数据库名称"

$env:PIT_STORAGE = "arcticdb"
$env:PIT_ARCTIC_URI = "lmdb://C:/nz/arcticdb?map_size=600GB"
$env:PIT_ARCTIC_LIBRARY = "factor_pit"
$env:PIT_START_DATE = "2018-01-01"
$env:PIT_END_DATE = "2026-07-28"

python .\download_new_pit.py
```

若改为Parquet，把 `PIT_STORAGE` 设为 `parquet`，并设置：

```powershell
$env:PIT_OUTPUT_DIR = "C:\Users\hyf\Desktop\因子\data\new_pit"
```

环境变量只在当前PowerShell窗口有效。不要把真实密码提交到GitHub。

## 数据口径

- 只保留合并报表：`MERGED_FLAG='1'`。
- 只保留年报、一季报、半年报、三季报：`A/Q1/S1/Q3`。
- 使用公告时点有效的上市和证券类型关系。
- 保留当前期、比较期、披露时间及后续修订。
- 保留 `PUBLISH_DATE`、`ACT_PUBTIME`、`UPDATE_TIME`。
- 添加 `IS_CURRENT_PERIOD`。

下载结果是公告事件表，不能按 `END_DATE` 提前回填。因子构建必须根据
`ACT_PUBTIME` 判断数据最早可用的交易日。

## 增量更新限制

正常向后更新可用 `resume=True`。如果供应商用 `UPDATE_TIME` 修改较早公告日的
历史记录，简单向后更新无法捕获；应定期用新的symbol或新Parquet目录完整重建并核对。

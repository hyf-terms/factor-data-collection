# 新准则三大报表PIT下载说明（Parquet版）

## 输出结构

程序从财务数据库读取三张表，直接按公告年份保存为分区Parquet，不再使用ArcticDB：

```text
data/new_pit/
├── new_pit_balance/
│   ├── year=2018/part-20180101-20181231.parquet
│   └── ...
├── new_pit_income/
│   └── year=2018/part-20180101-20181231.parquet
└── new_pit_cashflow/
    └── year=2018/part-20180101-20181231.parquet
```

| 报表 | 数据源 | Parquet数据集 |
|---|---|---|
| 资产负债表 | `vw_fdmt_bs_new` | `new_pit_balance` |
| 利润表 | `vw_fdmt_is_new` | `new_pit_income` |
| 现金流量表 | `vw_fdmt_cf_new` | `new_pit_cashflow` |

## 推荐：在Notebook运行

确认当前Python环境可以导入 `pandas`、`pyarrow` 和 `MySQLdb`；如缺少依赖：

```powershell
python -m pip install -r requirements.txt
```

先运行 `history_data.ipynb` 中创建数据库连接 `conn` 的单元格。无需创建
ArcticDB的 `ac` 或 `lib`：

```python
from download_new_pit import download_all_new_pit

summary = download_all_new_pit(
    conn=conn,
    output_dir=r"C:\Users\hyf\Desktop\因子\data\new_pit",
    start_date="2018-01-01",
    end_date="2026-07-27",
    resume=True,
)
summary
```

程序按公告年份逐块读取。每一块先完整写入临时文件，成功后再原子改名，避免中断后
留下伪装成完整数据的Parquet。`resume=True` 会扫描已有文件的最大
`PUBLISH_DATE`，从下一日继续。

## PowerShell命令行运行

在当前PowerShell窗口中设置环境变量：

```powershell
cd "C:\Users\hyf\Desktop\因子"

$env:PIT_DB_HOST = "数据库服务器地址"
$env:PIT_DB_PORT = "3306"
$env:PIT_DB_USER = "数据库用户名"
$env:PIT_DB_PASSWORD = "数据库密码"
$env:PIT_DB_NAME = "数据库名称"
$env:PIT_OUTPUT_DIR = "C:\Users\hyf\Desktop\因子\data\new_pit"
$env:PIT_START_DATE = "2018-01-01"
$env:PIT_END_DATE = "2026-07-27"

python .\download_new_pit.py
```

这些变量只在当前PowerShell窗口有效。不要把真实数据库密码写入代码、说明文件或
提交到GitHub。`PIT_OUTPUT_DIR` 未设置时，程序默认写入脚本旁的
`data\new_pit`。

## 数据口径

- 只保留合并报表：`MERGED_FLAG='1'`。
- 只保留年报、一季报、半年报、三季报：`A/Q1/S1/Q3`。
- 使用证券在公告时点有效的上市及证券类型关系。
- 保留当前期、比较期和后续修订记录。
- 保留 `PUBLISH_DATE`、`ACT_PUBTIME`、`UPDATE_TIME` 等PIT字段。
- 添加 `IS_CURRENT_PERIOD`，区分当前期与比较期。
- Parquet使用Zstandard压缩，日期、整数和财务数值统一数据类型。

下载数据仍是公告事件表，不能按 `END_DATE` 直接向历史日期回填。因子必须根据
`ACT_PUBTIME` 判断最早可用交易日。

## 检查和读取

```python
from download_new_pit import read_pit_dataset

income = read_pit_dataset(
    "new_pit_income",
    output_dir=r"C:\Users\hyf\Desktop\因子\data\new_pit",
)

income.shape, income["PUBLISH_DATE"].min(), income["PUBLISH_DATE"].max()
```

也可以直接使用Pandas读取整个分区目录：

```python
import pandas as pd

income = pd.read_parquet(
    r"C:\Users\hyf\Desktop\因子\data\new_pit\new_pit_income"
)
```

## 增量更新限制

断点续传适用于首次历史下载和正常向后更新。如果供应商通过 `UPDATE_TIME` 回填或
修改了较早公告日的记录，向后续传无法发现它。建议定期下载到一个新的空目录进行
完整重建，再核对行数和事件主键；不要直接覆盖唯一的数据副本。

# 新准则三大报表PIT本地下载说明

程序只执行以下流程：

```text
源MySQL数据库（只读）
        ↓
按公告年份查询PIT记录
        ↓
本地分区Parquet文件
```

不创建、不连接ArcticDB，也不依赖 `history_data.ipynb`。

## 本地输出

默认保存到：

```text
C:\Users\hyf\Desktop\因子\data\new_pit\
├── new_pit_balance\
│   ├── year=2018\part-20180101-20181231.parquet
│   └── ...
├── new_pit_income\
│   └── year=2018\part-20180101-20181231.parquet
└── new_pit_cashflow\
    └── year=2018\part-20180101-20181231.parquet
```

按年份分区可以避免单个文件过大，也便于断点续传。三个数据集目录可直接传给
`pandas.read_parquet()`。

## 第一次配置

安装依赖：

```powershell
cd "C:\Users\hyf\Desktop\因子"
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

复制配置模板：

```powershell
Copy-Item .\new_pit.env.example .\.env
```

使用文本编辑器打开 `.env`，填写真实的源数据库只读连接信息：

```text
PIT_DB_HOST=数据库服务器地址
PIT_DB_PORT=3306
PIT_DB_USER=数据库用户名
PIT_DB_PASSWORD=数据库密码
PIT_DB_NAME=数据库名称
PIT_OUTPUT_DIR=C:/Users/hyf/Desktop/因子/data/new_pit
PIT_START_DATE=2018-01-01
PIT_END_DATE=2026-07-28
```

`.env` 已被 `.gitignore` 排除，不应上传GitHub或发给他人。程序启动时会自动读取
脚本旁的 `.env`，不需要每次在PowerShell重新设置变量。

## 运行

无需激活虚拟环境：

```powershell
cd "C:\Users\hyf\Desktop\因子"
& ".\.venv\Scripts\python.exe" .\download_new_pit.py
```

程序依次下载：

| 数据源 | 本地数据集 |
|---|---|
| `vw_fdmt_bs_new` | `new_pit_balance` |
| `vw_fdmt_is_new` | `new_pit_income` |
| `vw_fdmt_cf_new` | `new_pit_cashflow` |

## 数据口径

- 只保留合并报表：`MERGED_FLAG='1'`。
- 只保留年报、一季报、半年报、三季报：`A/Q1/S1/Q3`。
- 使用公告时点有效的上市和证券类型关系。
- 保留当前期、比较期、披露时间和后续修订。
- 保留 `PUBLISH_DATE`、`ACT_PUBTIME`、`UPDATE_TIME`。
- 添加 `IS_CURRENT_PERIOD`。
- 使用PyArrow及Zstandard压缩写入Parquet。

## 读取与检查

```python
import pandas as pd

income = pd.read_parquet(
    r"C:\Users\hyf\Desktop\因子\data\new_pit\new_pit_income"
)

print(income.shape)
print(income["PUBLISH_DATE"].min(), income["PUBLISH_DATE"].max())
```

下载结果是公告事件表，不能按 `END_DATE` 提前回填。因子构建必须根据
`ACT_PUBTIME` 判断最早可用交易日。

## 断点续传

程序会读取已有Parquet的最大 `PUBLISH_DATE`，从下一日继续。若供应商后来修改了
更早公告日的数据，向后续传无法捕获；此时应换一个空目录完整重建并核对。

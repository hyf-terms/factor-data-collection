"""
数据查询 API 函数集合
"""
import pandas as pd
import datetime


def is_suspended(security_ids, trade_date, df_halt, conn):
    """查询指定股票在指定交易日的停牌状态
    
    Args:
        security_ids: 证券ID，单个整数或列表
        trade_date: 交易日期字符串，如 '2010-01-20'
        conn: 数据库连接对象
    
    Returns:
        DataFrame，包含 SECURITY_ID 和 IS_SUSPENDED 两列
    """
    # 统一处理为列表
    if isinstance(security_ids, (int, float)):
        security_ids = [int(security_ids)]
    else:
        security_ids = [int(sid) for sid in security_ids]
    
    td = pd.to_datetime(trade_date).date()
    market_open = datetime.time(9, 30)
    
    # 停牌生效：当天9:30或之前停牌才算当天生效
    halt_effective = (
        (df_halt['HALT_DATE'] < td) |
        ((df_halt['HALT_DATE'] == td) & (df_halt['HALT_TIME'] <= market_open))
    )
    
    # 复牌生效：当天9:30或之前复牌才算当天复牌
    resumed = (
        df_halt['RESUMP_BEGIN_TIME'].notna() &
        (
            (df_halt['RESUMP_DATE'] < td) |
            ((df_halt['RESUMP_DATE'] == td) & (df_halt['RESUMP_TIME'] <= market_open))
        )
    )
    
    # 停牌判断：停牌已生效且未复牌
    halted_set = set(df_halt.loc[halt_effective & (~resumed), 'SECURITY_ID'])

    # 返回结果
    return pd.DataFrame({
        'SECURITY_ID': security_ids,
        'IS_SUSPENDED': [sid in halted_set for sid in security_ids]
    })



def is_st(security_ids, trade_date, df_st_all, conn=None):
    """查询指定股票在指定交易日的ST状态（无未来函数）
    
    逻辑：
    - 筛选 EFF_DATE <= trade_date 的记录（只看已生效的状态变更）
    - 每只股票取最近一条记录，判断 PARTY_STATE 是否为 2（实施ST）
    
    PARTY_STATE 含义：
        1-撤销ST, 2-实施ST, 3-恢复上市, 4-暂停上市, 5-退市,
        6-进入退市整理期交易首日, 7-退市整理期最后交易日
    
    Args:
        security_ids: 证券ID，单个整数或列表
        trade_date: 交易日期字符串，如 '2010-01-20'
        df_st_all: 预加载的全量 equ_inst_sstate 数据（需含 SECURITY_ID, PARTY_STATE, EFF_DATE）
        conn: 数据库连接对象（保留接口一致性）
    
    Returns:
        DataFrame，包含 SECURITY_ID 和 IS_ST 两列
    """
    if isinstance(security_ids, (int, float)):
        security_ids = [int(security_ids)]
    else:
        security_ids = [int(sid) for sid in security_ids]
    
    td = pd.to_datetime(trade_date)
    
    # 只看已生效的记录，避免未来函数
    df_valid = df_st_all.loc[df_st_all['EFF_DATE'] <= td]
    
    # 每只股票取最近一条状态记录
    df_latest = df_valid.sort_values('EFF_DATE').groupby('SECURITY_ID').tail(1)
    
    # 只有 PARTY_STATE==1（撤销ST）才表示恢复正常，其余 2-7 均为异常状态
    st_set = set(df_latest.loc[df_latest['PARTY_STATE'] != 1, 'SECURITY_ID'])
    
    return pd.DataFrame({
        'SECURITY_ID': security_ids,
        'IS_ST': [sid in st_set for sid in security_ids]
    })



def get_citic_industry(security_ids, trade_date, df_citic_inst, df_citic_type, conn=None):
    """查询指定股票在指定交易日的中信行业分类（1/2/3级，无未来函数）

    Args:
        security_ids: 证券ID，单个整数或列表
        trade_date: 交易日期字符串
        df_citic_inst: 预加载（md_inst_type JOIN md_security, TYPE_ID LIKE '010317%'）
                       需含 SECURITY_ID, TYPE_ID, INTO_DATE, OUT_DATE
        df_citic_type: 预加载（md_type WHERE TYPE_ID LIKE '010317%'）
                       需含 TYPE_ID, TYPE_NAME
        conn: 保留接口一致性

    Returns:
        DataFrame，包含 SECURITY_ID 及中信行业1/2/3级代码和名称
    """
    if isinstance(security_ids, (int, float)):
        security_ids = [int(security_ids)]
    else:
        security_ids = [int(sid) for sid in security_ids]

    td = pd.to_datetime(trade_date)

    # 筛选 trade_date 生效的记录（避免未来函数）
    mask = (
        df_citic_inst['SECURITY_ID'].isin(security_ids) &
        (df_citic_inst['INTO_DATE'] <= td) &
        (df_citic_inst['OUT_DATE'].isna() | (df_citic_inst['OUT_DATE'] > td))
    )
    df_v = df_citic_inst.loc[mask].copy()

    # 同一股票多条记录时取 INTO_DATE 最近的
    df_v = df_v.sort_values('INTO_DATE').groupby('SECURITY_ID').tail(1)

    # 截取1/2/3级行业代码（前8/10/12位）
    df_v['CITIC_ID_1ST'] = df_v['TYPE_ID'].str[:8]
    df_v['CITIC_ID_2ND'] = df_v['TYPE_ID'].str[:10]
    df_v['CITIC_ID_3RD'] = df_v['TYPE_ID']

    # 名称映射
    nm = df_citic_type.set_index('TYPE_ID')['TYPE_NAME'].to_dict()
    df_v['CITIC_NAME_1ST'] = df_v['CITIC_ID_1ST'].map(nm)
    df_v['CITIC_NAME_2ND'] = df_v['CITIC_ID_2ND'].map(nm)
    df_v['CITIC_NAME_3RD'] = df_v['CITIC_ID_3RD'].map(nm)

    # 确保所有输入 security_ids 都有记录
    result = pd.DataFrame({'SECURITY_ID': security_ids})
    cols = ['SECURITY_ID', 'CITIC_ID_1ST', 'CITIC_NAME_1ST',
            'CITIC_ID_2ND', 'CITIC_NAME_2ND',
            'CITIC_ID_3RD', 'CITIC_NAME_3RD']
    return result.merge(df_v[cols], on='SECURITY_ID', how='left')

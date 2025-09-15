# ---encoding:utf-8---
# @Date: 2025/8/25
import os
import numpy as np
import pandas as pd

os.makedirs("data_clean", exist_ok=True)


def data_clean_8():
    df = pd.read_excel('data_raw/8.Quarterly - 1965 - GREIX_all_cities_data.xlsx')

    filtered_df = df[
        (df['City'] == 'GREIX') &
        (df['Property_Type'] == 'Mehrfamilienhäuser') &
        (df['Inflation'] == 0)
        ]

    df_1980_plus = filtered_df[filtered_df['Year'] >= 1980].copy()

    result_df = pd.DataFrame(columns=['Year', 'Quarter', 'Index', 'Last_Year_Index'])

    years = sorted(df_1980_plus['Year'].unique())

    for i in range(len(years) - 1):
        current_year = years[i]
        next_year = years[i + 1]

        current_year_null = df_1980_plus[
            (df_1980_plus['Year'] == current_year) &
            (df_1980_plus['Quarter'].isna())
            ]

        next_year_data = df_1980_plus[
            (df_1980_plus['Year'] == next_year) &
            (df_1980_plus['Quarter'].isin([1, 2, 3, 4]))
            ].copy()

        if not current_year_null.empty and not next_year_data.empty:
            last_index_value = current_year_null['Index'].values[0]
            next_year_data['Last_Year_Index'] = last_index_value
            next_year_data = next_year_data[['Year', 'Quarter', 'Index', 'Last_Year_Index']]
            result_df = pd.concat([result_df, next_year_data], ignore_index=True)

    result_df.to_excel('data_clean/8_data.xlsx', index=False)

    print("数据处理完成并已保存到 data_clean/8_data.xlsx")
    print(f"处理了 {len(result_df)} 行数据")


def data_clean_13():
    quarterly_df = pd.read_excel('data_clean/8_data.xlsx')
    macro_df = pd.read_excel('data_raw/13.Yearly -1965- Historical German Data (43 indicators).xlsx')

    # 筛选德国数据
    germany_macro = macro_df[macro_df['country'] == 'Germany'].copy()

    # 定义不同类型的指标处理方式
    STOCK_VARIABLES = ['pop', 'debtgdp', 'lev', 'ltd']  # 存量指标：简单重复
    FLOW_VARIABLES = ['gdp', 'imports', 'exports', 'revenue', 'expenditure']  # 流量指标：按季度分配
    RATE_VARIABLES = ['unemp', 'stir', 'ltrate', 'wage', 'cpi']  # 比率指标：简单重复

    selected_columns = ['year'] + STOCK_VARIABLES + FLOW_VARIABLES + RATE_VARIABLES
    macro_selected = germany_macro[selected_columns].copy()

    # 重命名列
    rename_dict = {col: f'ann_{col}' for col in macro_selected.columns if col != 'year'}
    rename_dict['year'] = 'year'
    macro_selected = macro_selected.rename(columns=rename_dict)

    # 合并数据
    enhanced_data = pd.DataFrame()

    for year in sorted(quarterly_df['Year'].unique()):
        year_quarters = quarterly_df[quarterly_df['Year'] == year].copy()
        prev_year_data = macro_selected[macro_selected['year'] == year - 1]

        if not prev_year_data.empty:
            # 处理存量指标和比率指标：重复
            for col in STOCK_VARIABLES + RATE_VARIABLES:
                macro_col = f'ann_{col}'
                if macro_col in prev_year_data.columns:
                    year_quarters[macro_col] = prev_year_data[macro_col].values[0]

            # 处理流量指标：按季度平均分配
            for col in FLOW_VARIABLES:
                macro_col = f'ann_{col}'
                if macro_col in prev_year_data.columns:
                    # 季度均匀分布
                    quarterly_value = prev_year_data[macro_col].values[0] / 4
                    year_quarters[macro_col] = quarterly_value

        enhanced_data = pd.concat([enhanced_data, year_quarters], ignore_index=True)

    enhanced_data.to_excel('data_clean/13_data.xlsx', index=False)
    return enhanced_data


def data_clean_13_all():
    quarterly_df = pd.read_excel('data_clean/8_data.xlsx')
    macro_df = pd.read_excel('data_raw/13.Yearly -1965- Historical German Data (43 indicators).xlsx')

    # 筛选德国数据
    germany_macro = macro_df[macro_df['country'] == 'Germany'].copy()

    # 定义基本列和不同类型的指标处理方式
    BASE_COLUMNS = ['year', 'country', 'iso', 'ifs']
    STOCK_VARIABLES = ['pop', 'debtgdp', 'lev', 'ltd']  # 存量指标：简单重复
    FLOW_VARIABLES = ['gdp', 'imports', 'exports', 'revenue', 'expenditure']  # 流量指标：按季度分配

    # 自动识别其他所有列作为比率指标
    all_columns = set(germany_macro.columns)
    used_columns = set(BASE_COLUMNS + STOCK_VARIABLES + FLOW_VARIABLES)
    RATE_VARIABLES = list(all_columns - used_columns)

    # 确保只保留数值型列且非空
    RATE_VARIABLES = [col for col in RATE_VARIABLES
                      if col not in BASE_COLUMNS
                      and col in germany_macro.select_dtypes(include='number').columns
                      and not germany_macro[col].isnull().all()]  # 新增：过滤全为空的列

    # 同样过滤STOCK和FLOW变量中的空列
    STOCK_VARIABLES = [col for col in STOCK_VARIABLES
                       if col in germany_macro.columns
                       and not germany_macro[col].isnull().all()]

    FLOW_VARIABLES = [col for col in FLOW_VARIABLES
                      if col in germany_macro.columns
                      and not germany_macro[col].isnull().all()]

    selected_columns = ['year'] + STOCK_VARIABLES + FLOW_VARIABLES + RATE_VARIABLES
    macro_selected = germany_macro[selected_columns].copy()

    # 重命名列
    rename_dict = {col: f'ann_{col}' for col in macro_selected.columns if col != 'year'}
    rename_dict['year'] = 'year'
    macro_selected = macro_selected.rename(columns=rename_dict)

    # 合并数据
    enhanced_data = pd.DataFrame()

    for year in sorted(quarterly_df['Year'].unique()):
        year_quarters = quarterly_df[quarterly_df['Year'] == year].copy()
        prev_year_data = macro_selected[macro_selected['year'] == year - 1]

        if not prev_year_data.empty:
            # 处理存量指标和比率指标：重复
            for col in STOCK_VARIABLES + RATE_VARIABLES:
                macro_col = f'ann_{col}'
                if macro_col in prev_year_data.columns:
                    year_quarters[macro_col] = prev_year_data[macro_col].values[0]

            # 处理流量指标：按季度平均分配
            for col in FLOW_VARIABLES:
                macro_col = f'ann_{col}'
                if macro_col in prev_year_data.columns:
                    # 季度均匀分布
                    quarterly_value = prev_year_data[macro_col].values[0] / 4
                    year_quarters[macro_col] = quarterly_value

        enhanced_data = pd.concat([enhanced_data, year_quarters], ignore_index=True)

    enhanced_data.to_excel('data_clean/13_data_all.xlsx', index=False)
    return enhanced_data

def data_clean_3():
    raw_file_path = 'data_raw/3.Daily -1976-  DAX & REITs.xlsx'
    clean_file_path = 'data_clean/10_data_all.xlsx'
    output_file_path = 'data_clean/3_data_all.xlsx'

    existing_quarterly_df = pd.read_excel(clean_file_path)

    tag_daily_df = pd.read_excel(raw_file_path, sheet_name='DAX')
    tag_daily_df['Date'] = pd.to_datetime(tag_daily_df['Date'])
    tag_daily_df.set_index('Date', inplace=True)
    tag_daily_df = tag_daily_df.sort_index()
    tag_daily_df['Daily_Return'] = tag_daily_df['Close'].pct_change()
    tag_daily_df['Daily_Range'] = (tag_daily_df['High'] - tag_daily_df['Low']) / tag_daily_df['Open']
    tag_daily_df['True_Range'] = np.maximum(
        tag_daily_df['High'] - tag_daily_df['Low'],
        np.maximum(
            abs(tag_daily_df['High'] - tag_daily_df['Close'].shift(1)),
            abs(tag_daily_df['Low'] - tag_daily_df['Close'].shift(1))
        )
    )
    # 重采样到季度频率，计算各种特征
    quarterly_features = tag_daily_df.resample('Q').agg({
        'Close': ['last', 'mean', 'std'],  # 季度末收盘价，季度均价，季度价标准差
        'High': 'max',  # 季度最高价
        'Low': 'min',  # 季度最低价
        'Open': 'first',  # 季度开盘价
        'Daily_Return': ['std', 'mean', 'skew'],  # 日收益率的标准差、均值、偏度
        'Daily_Range': 'mean',  # 平均日内波动幅度
        'True_Range': 'mean',  # 平均真实波动幅度
        'Volume': ['sum', 'mean']  # 季度总成交量和平均成交量
    })
    # 年化波动率 (年252个交易日)
    quarterly_features['Volatility'] = quarterly_features[('Daily_Return', 'std')] * np.sqrt(252)
    # 季度回报率
    quarterly_features['Quarterly_Return'] = quarterly_features[('Close', 'last')].pct_change()
    # 季度内最大回撤
    quarterly_features['Max_Drawdown'] = (
            (quarterly_features[('High', 'max')] - quarterly_features[('Low', 'min')]) /
            quarterly_features[('High', 'max')]
    )
    # 波动率比率（波动率/平均波动率）
    volatility_mean = quarterly_features['Volatility'].rolling(window=8, min_periods=1).mean()
    quarterly_features['Volatility_Ratio'] = quarterly_features['Volatility'] / volatility_mean
    # 清理列名
    quarterly_features.columns = ['_'.join(col).strip() for col in quarterly_features.columns.values]
    column_mapping = {}
    for col in quarterly_features.columns:
        if col.startswith('Close_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('High_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('Low_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('Open_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('Daily_Return_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('Daily_Range_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('True_Range_'):
            column_mapping[col] = f'TAG_{col}'
        elif col.startswith('Volume_'):
            column_mapping[col] = f'TAG_{col}'
        else:
            column_mapping[col] = f'TAG_{col}'

    quarterly_features.rename(columns=column_mapping, inplace=True)

    quarterly_features.reset_index(inplace=True)
    quarterly_features['Year'] = quarterly_features['Date'].dt.year
    quarterly_features['Quarter'] = quarterly_features['Date'].dt.quarter

    # 选择要保留的列（去除Date列）
    keep_columns = ['Year', 'Quarter'] + [col for col in quarterly_features.columns
                                          if col not in ['Date', 'Year', 'Quarter']]

    tag_quarterly_features = quarterly_features[keep_columns].copy()
    existing_quarterly_df['Year'] = existing_quarterly_df['Year'].astype(int)
    existing_quarterly_df['Quarter'] = existing_quarterly_df['Quarter'].astype(int)
    tag_quarterly_features['Year'] = tag_quarterly_features['Year'].astype(int)
    tag_quarterly_features['Quarter'] = tag_quarterly_features['Quarter'].astype(int)

    # 合并数据
    merged_df = existing_quarterly_df.merge(
        tag_quarterly_features,
        on=['Year', 'Quarter'],
        how='left'
    )

    # 对新增的数值列进行向前填充
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Quarter']]
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(method='ffill')

    merged_df.to_excel(output_file_path, index=False)

    return merged_df


if __name__ == '__main__':
    # data_clean_8()
    # data_clean_13()
    # data_clean_13_all()
    # 数据10手动复制过来
    data_clean_3()

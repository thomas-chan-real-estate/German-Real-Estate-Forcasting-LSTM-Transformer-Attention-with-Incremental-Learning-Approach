
## 数据说明

| 表头                  | 含义              | 理由          |
|-----------------------|-----------------|-------------|
| Year                  | 年份              | 时间维度标识      |
| Quarter               | 季度（1-4）         | 时间维度标识      |
| Index                 | 当期房地产指数(GREIX)值 | 目标变量        |
| Last_Year_Index       | 上年同期房地产指数值      | 基准对比        |
| ann_pop               | 年度人口总数          | 需求端长期驱动因素   |
| ann_debtgdp           | 政府债务占GDP比率      | 财政健康度指标     |
| ann_lev               | 银行杠杆率           | 金融体系风险指标    |
| ann_ltd               | 存贷比             | 银行信贷扩张能力    |
| ann_unemp             | 失业率             | 经济健康度同步指标   |
| ann_stir              | 短期利率            | 融资成本        |
| ann_ltrate            | 长期利率            | 抵押贷款利率基准    |
| ann_wage              | 工资指数            | 购买力指标       |
| ann_cpi               | 消费者价格指数         | 通胀压力，影响央行政策 |
| ann_gdp               | 国内生产总值          | 整体经济活力      |
| ann_imports           | 进口总额            | 对外贸易状况      |
| ann_exports           | 出口总额            | 对外贸易状况      |
| ann_revenue           | 政府财政收入          | 财政政策空间      |
| ann_expenditure       | 政府财政支出          | 财政政策力度      |
| contruction           | 建筑业指标           | 供给端活动指标     |
| TAG_Close_last        | 季度末收盘价          | 期末估值水平      |
| TAG_Close_mean        | 季度平均收盘价         | 平均估值水平      |
| TAG_Close_std         | 收盘价标准差          | 价格离散程度      |
| TAG_High_max          | 季度最高价           | 价格阻力位       |
| TAG_Low_min           | 季度最低价           | 价格支撑位       |
| TAG_Open_first        | 季度开盘价           | 期初估值水平      |
| TAG_Daily_Return_std  | 日收益率标准差         | 日频波动性       |
| TAG_Daily_Return_mean | 平均日收益率          | 日均收益        |
| TAG_Daily_Return_skew | 日收益率偏度          | 极端风险指标      |
| TAG_Daily_Range_mean  | 平均日内波动幅度        | 交易活跃度       |
| TAG_True_Range_mean   | 平均真实波动幅度        | 波动性质量指标     |
| TAG_Volume_sum        | 总成交量            | 市场流动性指标     |
| TAG_Volume_mean       | 平均成交量           | 日均流动性       |
| TAG_Volatility_       | 年化波动率           | 风险溢价指标      |
| TAG_Quarterly_Return_ | 季度回报率           | 极其重要的领先指标   |
| TAG_Max_Drawdown_     | 最大回撤            | 下行风险指标      |
| TAG_Volatility_Ratio_ | 波动率比率           | 波动性异常指标     |


## 创新点
### 创新点一：动态自适应特征选择机制
引入可学习的动态特征选择器：在训练阶段，采用Gumbel-Softmax技巧，端到端地优化特征选择策略； 动态适应性，能够根据不同时期数据分布的变化动态地筛选出最相关。

### 创新点二：融合架构与增量式滚动预测框架
构建LSTM-Transformer-Attention的深度学习架构，并将其与增量学习框架相结合。

## 算法流程

### stage1 数据预处理与特征工程
1. 数据加载与清洗：
    - 从指定的Excel文件读取数据。
    - 删除包含空值的行。
    - 按年份和季度排序，并创建时间索引。

2. 特征工程：
    - 技术指标：计算指数的增长率、动量和波动率。
    - 经济指标：计算GDP增长率、通胀压力、贸易差额和财政收支差额。
    - 金融市场指标：计算利差和信贷风险（债务与杠杆率的乘积）。
    - 删除工程后产生空值的行。

### stage2 数据标准化与序列构建
1. 特征/目标分离：将处理好的DataFrame分离为特征矩阵X和目标向量y（房地产指数）。

2. 数据标准化： 对特征X使用RobustScaler进行标准化。 对目标y使用StandardScaler进行标准化。

3. 构建时间序列： 使用滑动窗口方法将数据转换为监督学习格式。生成最终的模型输入X_seq和对应的标签y_seq。

### stage3 训练与预测循环
1. 划分训练/测试数据集：以2010年作为测试集的起始年份，找到其在序列数据中的对应索引test_start_idx。

2. 初始化模型：根据输入特征的维度input_dim初始化EnhancedLSTM神经网络。

3. 初始训练集：使用2010年之前的所有序列数据（X_seq[:test_start_idx]）作为初始训练集。

4. 模型训练：
    - 模型训练：将当前训练集数据转换为PyTorch Tensor并送入cuda
    - 使用AdamW优化器和ReduceLROnPlateau学习率调度器进行训练
    - 损失函数为均方误差
    - 采用梯度裁剪防止梯度爆炸
    - 采用早停机制
5. 模型预测:
    - 使用当前时间点的序列数据（X_seq[i]）进行预测，得到标准化后的预测值。
    - 通过scaler_y.inverse_transform将预测值反标准化为原始尺度。
    - 记录预测值、真实值和当前的特征重要性掩码。


### stage4 结果评估与分析

1. calculate_metrics:在测试集（2010年后）上计算MAE, MSE, RMSE, R², MAPE, 方向准确率等一系列指标。
2. _plot_results:主预测图,散点图,残差图,残差分布直方图
3. _analyze_feature_importance:动态特征选择器选择各个特征的平均重要性
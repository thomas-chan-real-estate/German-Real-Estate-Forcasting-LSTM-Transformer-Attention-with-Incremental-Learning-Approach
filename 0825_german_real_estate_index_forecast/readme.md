
## 数据说明

| Header                | Meaning                          | Rationale                           |
|-----------------------|----------------------------------|-------------------------------------|
| Year                  | Year                             | Time dimension indicator             |
| Quarter               | Quarter (1–4)                    | Time dimension indicator             |
| Index                 | Real Estate Index (GREIX) value  | Target variable                      |
| Last_Year_Index       | Real estate index of same quarter last year | Benchmark comparison       |
| ann_pop               | Annual population                | Long-term demand driver              |
| ann_debtgdp           | Government debt-to-GDP ratio     | Fiscal health indicator              |
| ann_lev               | Bank leverage ratio              | Financial system risk indicator      |
| ann_ltd               | Loan-to-deposit ratio            | Bank credit expansion capacity       |
| ann_unemp             | Unemployment rate                | Economic health indicator            |
| ann_stir              | Short-term interest rate         | Financing cost                       |
| ann_ltrate            | Long-term interest rate          | Mortgage benchmark                   |
| ann_wage              | Wage index                       | Purchasing power indicator           |
| ann_cpi               | Consumer Price Index             | Inflation pressure, affects central bank policy |
| ann_gdp               | Gross Domestic Product           | Overall economic vitality            |
| ann_imports           | Total imports                    | Trade status                         |
| ann_exports           | Total exports                    | Trade status                         |
| ann_revenue           | Government fiscal revenue        | Fiscal policy space                  |
| ann_expenditure       | Government fiscal expenditure    | Fiscal policy intensity              |
| contruction           | Construction indicator           | Supply-side activity indicator       |
| TAG_Close_last        | Quarter-end closing price        | End-of-period valuation              |
| TAG_Close_mean        | Average quarterly closing price  | Average valuation                    |
| TAG_Close_std         | Standard deviation of closing price | Price dispersion                   |
| TAG_High_max          | Quarterly maximum price          | Resistance level                     |
| TAG_Low_min           | Quarterly minimum price          | Support level                        |
| TAG_Open_first        | Opening price of quarter         | Initial valuation                    |
| TAG_Daily_Return_std  | Std. dev. of daily returns       | Daily volatility                     |
| TAG_Daily_Return_mean | Average daily return             | Daily average return                 |
| TAG_Daily_Return_skew | Skewness of daily returns        | Extreme risk indicator               |
| TAG_Daily_Range_mean  | Average intraday range           | Trading activity                     |
| TAG_True_Range_mean   | Average true range               | Volatility quality indicator         |
| TAG_Volume_sum        | Total trading volume             | Market liquidity indicator           |
| TAG_Volume_mean       | Average trading volume           | Daily liquidity                      |
| TAG_Volatility_       | Annualized volatility            | Risk premium indicator               |
| TAG_Quarterly_Return_ | Quarterly return                 | Key leading indicator                |
| TAG_Max_Drawdown_     | Maximum drawdown                 | Downside risk indicator              |
| TAG_Volatility_Ratio_ | Volatility ratio                 | Abnormal volatility indicator        |


## 创新点
### Innovation 1: Dynamic Adaptive Feature Selection Mechanism

Introduce a learnable dynamic feature selector: during the training stage, adopt the Gumbel-Softmax technique to optimize the feature selection strategy end-to-end; dynamic adaptability enables filtering out the most relevant features according to the changing distribution of data across different periods.

### Innovation 2: Fusion Architecture and Incremental Rolling Forecasting Framework

Construct a deep learning architecture of LSTM-Transformer-Attention and combine it with an incremental learning framework.

## Algorithm Workflow
### stage1 Data Preprocessing and Feature Engineering

Data loading and cleaning:

 - Read data from the specified Excel file.

 - Delete rows containing missing values.

 - Sort by year and quarter, and create a time index.

Feature engineering:

 - Technical indicators: calculate growth rate, momentum, and volatility of the index.

 - Economic indicators: calculate GDP growth rate, inflation pressure, trade balance, and fiscal balance.

 - Financial market indicators: calculate interest rate spread and credit risk (product of debt and leverage ratio).

 - Delete rows that generate missing values after engineering.

### stage2 Data Standardization and Sequence Construction

Feature/target separation: split the processed DataFrame into a feature matrix X and a target vector y (real estate index).

Data standardization: use RobustScaler for features X. Use StandardScaler for target y.

Sequence construction: use the sliding window method to transform data into a supervised learning format. Generate the final model input X_seq and corresponding label y_seq.

### stage3 Training and Prediction Loop

1. Train/test split: take 2010 as the starting year of the test set and find its corresponding index test_start_idx in the sequence data.

2. Model initialization: initialize the EnhancedLSTM neural network according to the input feature dimension input_dim.

3. Initial training set: use all sequence data before 2010 (X_seq[:test_start_idx]) as the initial training set.

4. Model training:

 -Model training: convert the current training set data into PyTorch Tensor and send to cuda.

 -Use AdamW optimizer and ReduceLROnPlateau learning rate scheduler for training.

 -Loss function is mean squared error.

 -Apply gradient clipping to prevent gradient explosion.

 -Apply early stopping mechanism.

5. Model prediction:

 -Use the sequence data at the current time point (X_seq[i]) for prediction to obtain the standardized predicted value.

 -Inverse transform the predicted value back to the original scale through scaler_y.inverse_transform.

 -Record the prediction, ground truth, and current feature importance mask.

### stage4 Results Evaluation and Analysis

calculate_metrics: calculate a series of metrics on the test set (post-2010), including MAE, MSE, RMSE, R², MAPE, and directional accuracy.

 _plot_results: main prediction plot, scatter plot, residual plot, residual distribution histogram.

_analyze_feature_importance: average importance of each feature selected by the dynamic feature selector.

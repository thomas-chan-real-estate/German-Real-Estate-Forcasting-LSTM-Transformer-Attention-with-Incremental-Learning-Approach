import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
from torch.nn import TransformerEncoder, TransformerEncoderLayer

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

save_dir = "output"
os.makedirs(save_dir, exist_ok=True)


class AttentionLayer(nn.Module):
    """注意力机制层"""

    def __init__(self, hidden_dim, use_pos_encoding=False):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_pos_encoding = use_pos_encoding
        # 注意力权重
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        # 可学习缩放因子
        self.scale = nn.Parameter(torch.ones(1))
        # 可选位置编码
        if self.use_pos_encoding:
            self.pos_encoding = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lstm_outputs):
        batch_size, seq_len, hidden_dim = lstm_outputs.shape

        # 加入位置编码
        if self.use_pos_encoding:
            pos = torch.arange(seq_len, device=lstm_outputs.device).float() / seq_len
            pos = pos.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, hidden_dim)
            lstm_outputs = lstm_outputs + self.pos_encoding(pos)

        # 计算注意力分数
        attention_scores = self.attention_weights(lstm_outputs) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 计算上下文向量，同时加入残差融合
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        context_vector = 0.8 * context_vector + 0.2 * lstm_outputs.mean(dim=1)

        return context_vector, attention_weights


class DynamicFeatureSelector(nn.Module):
    """动态特征选择器"""

    def __init__(self, input_dim, selected_features=15):
        super(DynamicFeatureSelector, self).__init__()
        self.input_dim = input_dim
        self.selected_features = min(selected_features, input_dim)
        self.feature_importance = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 32)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(input_dim // 2, 32), input_dim),
            nn.Sigmoid()
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        self.feature_projection = nn.Linear(input_dim, self.selected_features)

    def forward(self, x):
        # 计算特征重要性 - 使用全局平均池化
        pooled_x = x.mean(dim=1)
        importance_scores = self.feature_importance(pooled_x)
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(importance_scores) + 1e-8) + 1e-8)
            logits = (torch.log(importance_scores + 1e-8) + gumbel_noise) / self.temperature
            soft_mask = torch.softmax(logits, dim=-1)
            _, top_k_indices = torch.topk(importance_scores, self.selected_features, dim=1)
            hard_mask = torch.zeros_like(importance_scores)
            hard_mask.scatter_(1, top_k_indices, 1.0)
            selection_mask = hard_mask - soft_mask.detach() + soft_mask
        else:
            # 推理时直接使用硬选择
            _, top_k_indices = torch.topk(importance_scores, self.selected_features, dim=1)
            selection_mask = torch.zeros_like(importance_scores)
            selection_mask.scatter_(1, top_k_indices, 1.0)

        # 使用加权投影 对每个时间步应用特征选择权重
        weighted_x = x * selection_mask.unsqueeze(1)
        selected_features = self.feature_projection(weighted_x)

        return selected_features, selection_mask


class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, selected_features=15):
        super(EnhancedLSTM, self).__init__()
        self.feature_selector = DynamicFeatureSelector(input_dim, selected_features)

        self.lstm = nn.LSTM(
            selected_features, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=4, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        self.attention = AttentionLayer(hidden_dim * 2)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        selected_x, feature_mask = self.feature_selector(x)
        lstm_out, _ = self.lstm(selected_x)
        trans_out = self.transformer(lstm_out)
        context_vector, attention_weights = self.attention(trans_out)
        output = self.output_layer(context_vector)
        return output, feature_mask, attention_weights


class RealEstatePredictor:
    def __init__(self, sequence_length=8, hidden_dim=64, num_layers=2, dropout=0.3, selected_features=15):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.selected_features = selected_features

        self.scaler_X = RobustScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_excel(file_path)
        df_clean = df.dropna()
        df_clean = df_clean.sort_values(['Year', 'Quarter']).reset_index(drop=True)
        df_clean['time_index'] = range(len(df_clean))
        df_clean['year_quarter'] = df_clean['Year'] + (df_clean['Quarter'] - 1) * 0.25
        return df_clean

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df_fe = df.copy()
        # 技术指标特征
        df_fe['index_growth_rate'] = df_fe['Index'].pct_change()
        df_fe['index_momentum'] = df_fe['Index'].rolling(4).mean() / df_fe['Index']
        df_fe['index_volatility'] = df_fe['Index'].rolling(4).std()

        # 经济指标特征
        df_fe['gdp_growth'] = df_fe['ann_gdp'].pct_change()
        df_fe['inflation_pressure'] = df_fe['ann_cpi'].rolling(4).mean()
        df_fe['trade_balance'] = df_fe['ann_exports'] - df_fe['ann_imports']
        df_fe['fiscal_balance'] = df_fe['ann_revenue'] - df_fe['ann_expenditure']

        # 金融市场特征
        df_fe['interest_spread'] = df_fe['ann_ltrate'] - df_fe['ann_stir']
        df_fe['credit_risk'] = df_fe['ann_ltd'] * df_fe['ann_lev']
        df_fe = df_fe.dropna().reset_index(drop=True)

        return df_fe

    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评价指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        # 方向准确率
        direction_accuracy = np.mean(np.sign(y_true[1:] - y_true[:-1]) ==
                                     np.sign(y_pred[1:] - y_pred[:-1])) * 100

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }

    def _train_model(self, X_train, y_train, epochs=200):
        """训练模型"""
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
        criterion = nn.MSELoss()

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs, _, _ = self.model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(loss)

            # 早停机制
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 30:
                break

        return best_loss

    def train_and_predict(self, file_path: str):
        """主训练和预测流程（保留原有增量训练）"""
        # 1. 数据加载和预处理
        df = self.load_and_preprocess_data(file_path)
        df = self.feature_engineering(df)

        # 2. 准备特征和目标变量
        feature_cols = [col for col in df.columns if col not in ['Year', 'Quarter', 'Index']]
        X = df[feature_cols].values
        y = df['Index'].values

        # 3. 数据标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 4. 创建序列数据
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        # 5. 分割数据
        test_start_year = 2010
        test_start_idx = df[df['Year'] >= test_start_year].index[0] - self.sequence_length + 5
        # test_start_idx = max(test_start_idx, len(X_seq) // 2)

        # 6. 初始化模型
        input_dim = X_seq.shape[2]
        self.model = EnhancedLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            selected_features=min(self.selected_features, input_dim)
        ).to(self.device)

        # 7. 增量训练和预测（原有逻辑）
        predictions_after2010 = []
        actual_values_after2010 = []
        feature_importance_history = []

        current_train_X = X_seq[:test_start_idx]
        current_train_y = y_seq[:test_start_idx]

        for i in range(test_start_idx, len(X_seq)):
            train_loss = self._train_model(current_train_X, current_train_y)

            # 预测下一个时间点
            test_X = torch.tensor(X_seq[i:i + 1], dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred_scaled, feature_mask, attention_weights = self.model(test_X)
                pred_scaled = pred_scaled.cpu().numpy().flatten()

            # 反标准化
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            actual = self.scaler_y.inverse_transform(y_seq[i:i + 1].reshape(-1, 1)).flatten()[0]

            predictions_after2010.append(pred)
            actual_values_after2010.append(actual)
            feature_importance_history.append(feature_mask.cpu().numpy())

            current_train_X = np.vstack([current_train_X, X_seq[i:i + 1]])
            current_train_y = np.hstack([current_train_y, y_seq[i:i + 1]])

            print(f"预测值: {pred:.4f}, 实际值: {actual:.4f}, 误差: {abs(pred - actual):.4f}")

        # 8. 训练完成后，用最终模型预测2010年前的数据
        predicted_before2010 = []
        for i in range(test_start_idx):
            test_X = torch.tensor(X_seq[i:i + 1], dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred_scaled, _, _ = self.model(test_X)
                pred = self.scaler_y.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()[0]
                predicted_before2010.append(pred)

        # 9. 对齐原始 df 长度
        predicted_full_aligned = np.full(len(df), np.nan)
        actual_full_aligned = np.full(len(df), np.nan)

        # 填充预测值
        predicted_full_aligned[self.sequence_length:] = np.array(predicted_before2010 + predictions_after2010)
        # 填充实际值
        actual_full_aligned[self.sequence_length:] = np.array(
            list(self.scaler_y.inverse_transform(
                y_seq[:test_start_idx].reshape(-1, 1)).flatten()) + actual_values_after2010
        )

        # 10. 计算测试集指标
        metrics = self.calculate_metrics(actual_full_aligned[test_start_idx:], predicted_full_aligned[test_start_idx:])
        print("\n=== 模型评价指标（测试集） ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # 11. 绘制全序列预测
        self._plot_results(actual_full_aligned, predicted_full_aligned, df, test_start_idx)

        # 12. 分析特征重要性
        self._analyze_feature_importance(feature_importance_history, feature_cols)

        return metrics, predicted_full_aligned, actual_full_aligned

    def _plot_results(self, actual, predicted_full, df, test_start_idx):
        """绘制预测结果及残差分析"""
        years = df['year_quarter'].values

        # --- 时间序列预测结果 ---
        plt.figure(figsize=(12, 6))
        plt.plot(years, actual, label='实际值', color='blue', linewidth=2)
        # 2010年前预测（虚线橙色）
        plt.plot(years[:test_start_idx], predicted_full[:test_start_idx],
                 label='预测值（2010年前）', color='orange', linestyle='--', linewidth=2)
        # 2010年后预测（实线红色）
        plt.plot(years[test_start_idx:], predicted_full[test_start_idx:],
                 label='预测值（2010年后）', color='red', linewidth=2, alpha=0.8)
        plt.title('房地产指数预测结果', fontsize=14, fontweight='bold')
        plt.xlabel('年份')
        plt.ylabel('指数值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "main_prediction.png"))
        plt.close()

        # --- 散点图 ---
        # 注意过滤掉 nan
        valid_idx = ~np.isnan(actual) & ~np.isnan(predicted_full)
        actual_valid = actual[valid_idx]
        predicted_valid = predicted_full[valid_idx]

        plt.figure(figsize=(6, 6))
        plt.scatter(actual_valid, predicted_valid, alpha=0.6, color='green')
        plt.plot([actual_valid.min(), actual_valid.max()],
                 [actual_valid.min(), actual_valid.max()], 'r--', lw=2)
        plt.xlabel('Actural Index')
        plt.ylabel('Predicted Index')
        plt.title('Actural vs Predicted')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "scatter.png"))
        plt.close()

        # --- 残差图 ---
        residuals = actual_valid - predicted_valid
        plt.figure(figsize=(12, 4))
        plt.plot(residuals, color='purple', linewidth=1.5)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('预测残差')
        plt.xlabel('时间步')
        plt.ylabel('残差')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "residuals.png"))
        plt.close()

        # --- 残差分布 ---
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title('残差分布')
        plt.xlabel('残差值')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "residuals_hist.png"))
        plt.close()

    def _analyze_feature_importance(self, feature_importance_history, feature_names):
        """分析特征重要性"""
        # 计算平均特征重要性
        avg_importance = np.mean(feature_importance_history, axis=0).flatten()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_importance
        }).sort_values('Importance', ascending=False)

        print("\n=== Top 10 most important features ===")
        print(importance_df.head(10).to_string(index=False))

        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)

        plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 15 most important features', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importance.png"))


if __name__ == "__main__":
    predictor = RealEstatePredictor(
        sequence_length=5,
        hidden_dim=8,
        num_layers=2,
        dropout=0.15,
        selected_features=25
    )

    # 执行训练和预测
    file_path = "data_clean/3_data_all.xlsx"
    metrics, predictions, actual_values = predictor.train_and_predict(file_path)

    print(f"Final R²: {metrics['R2']:.4f}")
    print(f"Directional Prediction Accuracy: {metrics['Direction_Accuracy']:.2f}%")

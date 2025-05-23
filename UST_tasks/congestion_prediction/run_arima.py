import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from utils.read_utils import load_json

# 假设你的数据是这样的：
data = load_json('./Data/Beijing_12h.json')

# 初始化保存误差
mse_list = []
mae_list = []
mape_list = []
accuracy_list = []

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero_indices = y_true != 0
    y_true = y_true[nonzero_indices]
    y_pred = y_pred[nonzero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true))

def cal_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    return correct_count / len(y_true)

def check_stationarity(series, significance_level=0.05):
    if np.all(series == series[0]):
        # 全部数值相同，常数序列，直接认定为平稳
        return True
    result = adfuller(series)
    p_value = result[1]
    return p_value < significance_level  # 返回 True/False

for idx, d in enumerate(tqdm(data)):
    recent = np.array([v for v, y in d['recent']])
    answer = np.array([v for v, y in d['answer']])

    # Step 1: 检测平稳性
    is_stationary = check_stationarity(recent)

    # 如果非平稳，先做一阶差分
    if not is_stationary:
        recent = np.diff(recent)
        # print(f"Sample {idx}: Series was non-stationary, performed differencing.")

    # Step 2: 用 Auto-ARIMA 拟合（因为我们手动差分了，所以设 d=0）
    try:
        model = auto_arima(
            recent,
            seasonal=False,
            suppress_warnings=True,
            error_action="ignore",
            d=0  # 禁止 auto_arima 自动再差分
        )
    except Exception as e:
        # print(f"Auto-ARIMA failed on sample {idx} with error: {e}")
        continue

    # Step 3: 预测未来 12 步
    n_forecast = len(answer)
    forecast = model.predict(n_periods=n_forecast)

    # Step 4: 如果做过差分，需要把预测值还原（累加）
    if not is_stationary:
        last_observed = [v for v, y in d['recent']][-1]
        forecast = np.cumsum(forecast) + last_observed
    forecast = np.clip(forecast, 0, 4)

    # Step 5: 计算误差
    mse = mean_squared_error(answer, forecast)
    mae = mean_absolute_error(answer, forecast)
    mape = mean_absolute_percentage_error(answer, forecast)
    accuracy = cal_accuracy(answer, forecast)

    mse_list.append(mse)
    mae_list.append(mae)
    mape_list.append(mape)
    accuracy_list.append(accuracy)

    # print(f"Sample {idx}: MSE={mse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

    # Step 6: 汇总平均误差
    print("\n=== Overall Metrics ===")
    print(f"Average MSE: {np.mean(mse_list):.4f}")
    print(f"Average MAE: {np.mean(mae_list):.4f}")
    print(f"Average MAPE: {np.mean(mape_list)*100:.2f}%")
    print(f"Average Accuracy: {np.mean(accuracy_list)*100:.2f}%")

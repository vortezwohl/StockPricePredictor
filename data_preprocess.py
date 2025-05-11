import torch
from torch.utils.data import TensorDataset, DataLoader

from data_reader import *


def create_autoregression_dataset(ticker_data_list: list[dict], n_past_days: int = 5):
    """
    为单个股票代码的时间序列数据创建自回归数据集。

    参数:
    ticker_data_list (list): 包含单个股票代码每日数据的字典列表。
    n_past_days (int): 用于预测的过去天数。

    返回:
    tuple: (X, Y) X是输入特征，Y是标签。
           X的形状是 (n_samples, n_past_days * n_features_per_day)
           Y的形状是 (n_samples, n_features_per_day)
    """

    def extract_daily_features(day_data: dict):
        open_price = day_data['Open']
        high_diff = day_data['High'] - open_price
        low_diff = day_data['Low'] - open_price
        close_diff = day_data['Adjusted'] - open_price
        volume = day_data['Volume']
        return [open_price, high_diff, low_diff, close_diff, volume]

    X, Y = [], []

    daily_features_list = []
    for day_data in ticker_data_list:
        # 确保所有必要字段都存在且为数字
        try:
            features = extract_daily_features(day_data)
            daily_features_list.append(features)
        except (TypeError, KeyError) as e:
            print(f"Skipping day due to missing or invalid data: {day_data.get('Date', 'Unknown Date')}, Error: {e}")
            continue

    # 2. 创建滑动窗口数据集
    # 我们需要 n_past_days 用于输入，1天用于标签，所以总共需要 n_past_days + 1 天的数据
    if len(daily_features_list) < n_past_days + 1:
        return torch.tensor([]), torch.tensor([])  # 不足以创建任何样本

    for i in range(len(daily_features_list) - n_past_days):
        # 输入特征：前 n_past_days 天的数据
        past_features = []
        for j in range(n_past_days):
            past_features.append(daily_features_list[i + j])
        X.append(past_features)

        # 标签：第 (n_past_days+1) 天的数据
        Y.append(daily_features_list[i + n_past_days])

    return torch.tensor(X), torch.tensor(Y)


_dataset = [{
    'ticker': x[0]['Ticker'],
    'data': create_autoregression_dataset(x, n_past_days=5)
}for x in dataset]


def get_dataset(ticker: str, train_ratio: float = 0.95, batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    _dataset_from_ticker = [x for x in _dataset if x['ticker'] == ticker][0]['data']
    inputs = torch.stack([x[0] for x in _dataset_from_ticker[0]])
    labels = torch.stack([x[1] for x in _dataset_from_ticker[1]])
    dataset_size = len(labels)
    train_size = int(train_ratio * dataset_size)
    train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
    valid_dataset = TensorDataset(inputs[train_size:], labels[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader

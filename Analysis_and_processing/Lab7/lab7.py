#%%
import pandas as pd
import numpy as np
import requests
import json
import torch.utils.data
from bs4 import BeautifulSoup
import re
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import itertools
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sympy.codegen.ast import Raise
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
import torch.optim as optim
import torch
from pathlib import Path
import optuna
#%%
# Parsing the website and getting the data
url = 'https://charts.finance.ua/ua/currency/data-archive?for=interbank&source=1&indicator=eur'
response = requests.get(url)
js = json.JSONDecoder().decode(response.text)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Turning list to DataFrame for more comfortable usage later and cutting to 2024
df = pd.DataFrame(js, columns=['Date', 'Buy', 'Sell'])
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', dayfirst=True)
df_full = df[(df['Date'] > '2022-08-01') & (df['Date'] < '2024-12-31')].reset_index(drop=True)
df_full[['Buy', 'Sell']] = df_full[['Buy', 'Sell']].astype(float)
df_1 = df_full[(df_full['Date'] > '2022-08-01') & (df_full['Date'] < '2024-08-01')].reset_index(drop=True)
df_1[['Buy', 'Sell']] = df_1[['Buy', 'Sell']].astype(float)
df = df_full[(df_full['Date'] > '2022-08-01') & (df_full['Date'] < '2024-01-01')].reset_index(drop=True)
df[['Buy', 'Sell']] = df[['Buy', 'Sell']].astype(float)

df_1
#%%
print(len(df), len(df_1), len(df_full)), device
#%%
def draw_plots(dtf, date_arg, second_arg):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # fig.tight_layout()

    # Common plot with basic information and EMA trend
    ax[0].set_xlabel('Графік реальних даних з накладеною EMA')
    ax[0].plot(dtf[date_arg], dtf[second_arg], color='g')
    ax[0].plot(dtf[date_arg], dtf[second_arg].ewm(span=50, adjust=False).mean(), color='b')

    # Histogram
    ax[1].hist(dtf[second_arg], bins=np.arange(round(dtf[second_arg].min()), round(dtf[second_arg].max() + 1), 0.5), edgecolor='black')
    ax[1].set_xlabel('Графік емпіричного розподілу частот')

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"Модель: {model_name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2: {r2:.4f}")
    return rmse, mape, r2
#%%
draw_plots(df_full, 'Date', 'Buy')
#%%
def prepare_regression_data(series, n_lags=10, n_future=1):
    X, y = [], []
    data = series.values

    for i in range(len(data) - n_lags - n_future + 1):
        X.append(data[i:i+n_lags])
        y.append(data[i+n_lags:(i+n_lags+n_future)])

    return np.array(X), np.array(y)

# def create_mlp_model(n_layers=1, units=32, activation='relu', input_shape=None):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#
#     for _ in range(n_layers):
#         model.add(Dense(units, activation=activation))
#
#     model.add(Dense(1, activation='linear'))
#
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def optimize_regression(X_train, y_train, input_shape):
#     keras_wrap = KerasRegressor(build_fn=create_mlp_model, input_shape=input_shape, verbose=0)
#
#     pipe = Pipeline([
#         ('scaler', MinMaxScaler()),
#         ('mlp', keras_wrap)
#     ])
#
#     param_grid = {
#         'mlp__n_layers': [1, 2, 3],
#         'mlp__units': [32, 64, 128],
#         'mlp__epochs': [50, 100],
#         'mlp__batch_size': [16, 32]
#     }
#     # tscv = TimeSeriesSplit(n_splits=5)
#     grid = GridSearchCV(
#         pipe,
#         param_grid,
#         # cv=tscv,
#         scoring='neg_root_mean_squared_error',
#         verbose=0,
#         n_jobs=-1
#     )
#     grid.fit(X_train, y_train)
#
#     print(f"Найкращі параметри Регресії: {grid.best_params_}")
#     return grid.best_estimator_, grid.score(X_train, y_train)

class MyDynamicRNN(nn.Module):
    def __init__(self, model_type: str, input_size: int, layer_sizes, dropouts, output_size=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList(nn.Dropout(p) for p in dropouts)
        self.model_type = model_type
        self.layer_sizes = layer_sizes
        self.output_size = output_size

        current_size = input_size
        for i, hidden_size in enumerate(layer_sizes):
            if model_type == 'LSTM':
                self.layers.append(nn.LSTM(current_size, hidden_size, batch_first=True))
            else:
                self.layers.append(nn.GRU(current_size, hidden_size, batch_first=True))

            current_size = hidden_size

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            x = self.dropouts[i](x)

        out = x[:, -1, :]
        out = self.fc(out)

        return out

def objective_rnn(trial, n_future_preds):
    rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU"])
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layer_sizes = []
    dropout_rates = []

    for i in range(n_layers):
        layer_sizes.append(trial.suggest_int(f"n_units_l{i}", 32, 128, step=16))
        dropout_rates.append(trial.suggest_float(f"dropout_l{i}", 0.2, 0.5, step=0.1))

    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = MyDynamicRNN(rnn_type, 1, layer_sizes, dropout_rates, n_future_preds).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_x_tensor = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
            if n_future_preds == 1:
                test_y_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)
            else:
                test_y_tensor = torch.FloatTensor(y_test).to(device)

            predictions = model(test_x_tensor)
            val_loss = criterion(predictions, test_y_tensor).item()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

def recursive_forecast(model, initial_lags, n_steps):
    current_features = initial_lags.to(device).view(1, -1).unsqueeze(-1)
    predictions = torch.tensor([], device=device)

    for _ in range(n_steps):
        pred = model(current_features)
        predictions = torch.concat([predictions, pred])
        current_features = torch.cat([current_features[:, :-1], pred.unsqueeze(-1),], dim=1)

    return predictions

def extrapolate_longrange(model, data, n_range_forward, n_steps_ahead):
    current_features = data[-1].to(device).view(1, -1).unsqueeze(-1)
    predictions = torch.tensor([], device=device)

    for _ in range(n_steps_ahead):
        pred = model(current_features)
        # print(pred)
        predictions = torch.concat([predictions, pred])
        # print(current_features[:, :-1])
        # print(pred.reshape(1, -1, 1))
        # print(pred.reshape(1, -1, 1)[:, 0].unsqueeze(-1))
        # print(torch.cat([current_features[:, :-1], pred.reshape(1, -1, 1)[:, 0].unsqueeze(-1)], dim=1))
        # current_features = torch.cat([current_features[:, :-1], pred.reshape(1, -1, 1)[:, 0].unsqueeze(-1)], dim=1)
        current_features = pred.reshape(1, -1, 1)

    return predictions

def show_extrapolaton_result(model, result_scaler, extrapolation_type=0, train_input_tensor=torch.tensor([]), input_test_tensor=torch.tensor([]), test_data_real=torch.tensor([]), test_size=10):
    rmse, mape, r2 = [None, None, None]
    model.eval()
    with torch.no_grad():
        if extrapolation_type == 0:
            if not len(train_input_tensor):
                raise Exception('train_input_tensor not specified')

            if model.output_size == 1:
                y_pred = recursive_forecast(model, train_input_tensor[-1].cpu(), test_size)
                y_pred_unscaled = result_scaler.inverse_transform(y_pred.cpu().numpy())
                y_pred_unscaled = y_pred_unscaled.reshape(-1, 1)
            else:
                y_pred = extrapolate_longrange(model, train_input_tensor, model.output_size, test_size)
                y_pred_unscaled = result_scaler.inverse_transform(y_pred.cpu().numpy())

            if len(test_data_real):
                rmse, mape, r2 = evaluate_model(test_data_real, y_pred_unscaled, model.model_type)

        else:
            if not len(input_test_tensor):
                raise Exception('input_test_tensor not specified')

            y_pred = model(input_test_tensor).cpu().numpy()
            y_pred_unscaled = result_scaler.inverse_transform(y_pred)

            if len(test_data_real):
                rmse, mape, r2 = evaluate_model(test_data_real, y_pred_unscaled, model.model_type)

    return rmse, mape, r2, y_pred_unscaled

def save_model(model: torch.nn.Module, model_name: str):
    model_path = Path("models")
    model_path.mkdir(parents=True,
                     exist_ok=True)

    model_full_name = f"{model_name}.pth"
    model_save_path = model_path / model_full_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
#%%
# lags_num = 10
lags_num = 25
n_future = 1
# n_future = 10

X, y = prepare_regression_data(df_full['Buy'], n_lags=lags_num, n_future=n_future)
X_train, X_test, y_train, y_test = (data for data in train_test_split(X, y, test_size=0.2, shuffle=False))

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_scaled = scaler_X.transform(X)

if n_future == 1:
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1))
else:
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    y_scaled = scaler_y.transform(y)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)

print(f"X_train shape: {X_train_tensor.shape}")
print(f"y_train shape: {y_train_tensor.shape}")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#%%
def train_rnn_model(n_future_preds, study, num_epochs=100):
    best_params = study.best_params
    best_rnn_type = best_params["rnn_type"]
    best_n_layers = best_params["n_layers"]
    best_layer_sizes = [best_params[f"n_units_l{i}"] for i in range(best_n_layers)]
    best_dropouts = [best_params[f"dropout_l{i}"] for i in range(best_n_layers)]

    model = MyDynamicRNN(best_rnn_type, 1, best_layer_sizes, best_dropouts, n_future_preds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    loss_history = []
    print(f"\nТренування моделі {best_rnn_type}:")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(train_loader))

    print('Тренування завершено')
    return model
#%%
best_params = {'batch_size': 16}
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=False)

storage_url = "sqlite:///lab7_optuna.db"

# study1 = optuna.create_study(
#     direction="minimize",
#     storage=storage_url,
#     study_name="time_series_rnn_v1",
#     load_if_exists=True,
#     pruner=optuna.pruners.MedianPruner(),
# )
#
# study1.optimize(lambda trial: objective_rnn(trial, n_future), n_trials=50)

# study2 = optuna.create_study(
#     direction="minimize",
#     storage=storage_url,
#     study_name="time_series_rnn_v2",
#     load_if_exists=True,
#     pruner=optuna.pruners.MedianPruner(),
# )
#
# study2.optimize(lambda trial: objective_rnn(trial, n_future), n_trials=50)

# study3 = optuna.create_study(
#     direction="minimize",
#     storage=storage_url,
#     study_name="time_series_rnn_v3",
#     load_if_exists=True,
#     pruner=optuna.pruners.MedianPruner(),
# )

study4 = optuna.create_study(
    direction="minimize",
    storage=storage_url,
    study_name="time_series_rnn_v4",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(),
)

study4.optimize(lambda trial: objective_rnn(trial, n_future), n_trials=50)

print('\nНайкращі знайдені параметри')
# print(study1.best_params)
# print(study2.best_params)
# print(study3.best_params)
print(study4.best_params)
#%%
# rnn_model_v1_1 = train_rnn_model(n_future, study1, num_epochs=50)
# rnn_model_v1_10 = train_rnn_model(n_future, study2, num_epochs=50)
# rnn_model_v2_25_10 = train_rnn_model(n_future, study3, num_epochs=50)
rnn_model_v2_25_1 = train_rnn_model(n_future, study4, num_epochs=50)

#%%
test_size = len(y_test)
# rmse, mape, r2, y_pred_unscaled = show_extrapolaton_result(model=rnn_model_v1_1, result_scaler=scaler_y, extrapolation_type=0, train_input_tensor=X_train_tensor, test_data_real=y_test, test_size=test_size)
# rmse, mape, r2, y_pred_unscaled = show_extrapolaton_result(model=rnn_model_v1_1, result_scaler=scaler_y, extrapolation_type=2, train_input_tensor=X_train_tensor, input_test_tensor=X_test_tensor, test_data_real=y_test, test_size=test_size)

# rmse, mape, r2, y_pred_unscaled = show_extrapolaton_result(model=rnn_model_v1_10, result_scaler=scaler_y, extrapolation_type=0, train_input_tensor=X_train_tensor, test_data_real=y_test, test_size=test_size)

rmse, mape, r2, y_pred_unscaled = show_extrapolaton_result(model=rnn_model_v2_25_1, result_scaler=scaler_y, extrapolation_type=2, train_input_tensor=X_train_tensor, input_test_tensor=X_test_tensor, test_data_real=y_test, test_size=test_size)

model_to_show = rnn_model_v1_1
# model_to_show = rnn_model_v2_25_10
if  n_future != 1:
    y_pred_unscaled = y_pred_unscaled[:, 0].reshape(-1, 1)
print(y_pred_unscaled.shape)
plt.figure(figsize=(12, 6))
plt.plot(df_full['Date'].iloc[-test_size:], df_full['Buy'].iloc[-test_size:], label='Реальні дані', color='black')
plt.plot(df_full['Date'].iloc[-test_size:], y_pred_unscaled, label=f'Прогноз {model_to_show.model_type}', color='red', linestyle='--')
plt.title(f'Прогноз валюти: Найкраща знайдена архітектура ({len(model_to_show.layer_sizes)} layers)\n моделі rnn_model_v2_25_1')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.grid(True)
plt.show()
#%%
test_size = len(y_test)
rmse, mape, r2, y_pred_unscaled = show_extrapolaton_result(model=rnn_model_v2_25_10, result_scaler=scaler_y, extrapolation_type=0, train_input_tensor=X_train_tensor, test_data_real=y_test, test_size=test_size)

# model_to_show = rnn_model_v1_1
model_to_show = rnn_model_v2_25_10
if  n_future != 1:
    y_pred_unscaled = y_pred_unscaled[:, 0].reshape(-1, 1)
print(y_pred_unscaled.shape)
plt.figure(figsize=(12, 6))
plt.plot(df_full['Date'].iloc[-test_size:], df_full['Buy'].iloc[-test_size:], label='Реальні дані', color='black')
plt.plot(df_full['Date'].iloc[-test_size:], y_pred_unscaled, label=f'Прогноз {model_to_show.model_type}', color='red', linestyle='--')
plt.title(f'Прогноз валюти: Найкраща знайдена архітектура ({len(model_to_show.layer_sizes)} layers)\n моделі rnn_model_v2_25_10')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.grid(True)
plt.show()
#%%
save_model(rnn_model_v1_1, 'RNN_model_v1_1')
# save_model(rnn_model_v1_10, 'RNN_model_v1_10')
#%%
test_size = len(y)
rmse, mape, r2, y_pred_unscaled = show_extrapolaton_result(model=rnn_model_v2_25_1, result_scaler=scaler_y, extrapolation_type=2, train_input_tensor=X_train_tensor, input_test_tensor=X_tensor, test_data_real=y, test_size=test_size)

# model_to_show = rnn_model_v1_1
model_to_show = rnn_model_v2_25_10
if  n_future != 1:
    y_pred_unscaled = y_pred_unscaled[:, 0].reshape(-1, 1)
print(y_pred_unscaled.shape)
plt.figure(figsize=(12, 6))
plt.plot(df_full['Date'].iloc[-test_size:], df_full['Buy'].iloc[-test_size:], label='Реальні дані', color='black')
plt.plot(df_full['Date'].iloc[-test_size:], y_pred_unscaled, label=f'Прогноз {model_to_show.model_type}', color='red', linestyle='--')
plt.title(f'Прогноз валюти: Найкраща знайдена архітектура ({len(model_to_show.layer_sizes)} layers)')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.grid(True)
plt.show()
#%%

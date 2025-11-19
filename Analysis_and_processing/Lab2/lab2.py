#%% md
# ## Завдання 1 та 2 рівня складності
# ### Група вимог_1:
# 1. Отримання вхідних даних із властивостями, заданими в Лр_1;
# 2. Визначення показників якості та оптимізація моделі (вибір моделі залежно від
# значення показника якості). Показник якості та спосіб оптимізації обрати самостійно.
# 3. Статистичне навчання поліноміальної моделі за методом найменших квадратів
# (МНК – LSM) – поліноміальна регресія для вхідних даних, отриманих в п.1,2. Спосіб
# реалізації МНК обрати самостійно;
# 4. Прогнозування (екстраполяцію) параметрів досліджуваного процесу за «навченою»
# у п.5 моделлю на 0,5 інтервалу спостереження (об’єму вибірки);
# 5. Провести аналіз отриманих результатів та верифікацію розробленого скрипта.
# 
# ### Група вимог_2:
# До функціоналу групи вимог 2 додати:
# 1. Модель вхідних даних із аномальними вимірами (якщо обрано синтезовані дані для
# обробки). Для реальних даних цей пункт пропустикт;
# 2. Очищення вхідних даних від аномальних вимірів. Спосіб виявлення аномалій та
# очищення обрати самостійно;
#%% md
# ### Проведення парсингу згідно з методом лабораторної роботи №1 та обробка аномальних даних (пункт 1 з групи вимог №1 та пункт 2 з групи вимог №2)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from sklearn.preprocessing import StandardScaler
import statsmodels.tsa.seasonal as seas
import statsmodels.api as sm
from bs4 import BeautifulSoup
from pathlib import Path
#%%
# Parsing the website and getting the data
url = 'https://charts.finance.ua/ua/currency/data-archive?for=interbank&source=1&indicator=eur'
response = requests.get(url)
js = json.JSONDecoder().decode(response.text)

# Turning list to DataFrame for more comfortable usage later and cutting to 2024
df = pd.DataFrame(js, columns=['Date', 'Buy', 'Sell'])
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', dayfirst=True)
df_full = df[(df['Date'] > '2021-12-31') & (df['Date'] < '2024-12-31')].reset_index(drop=True)
df_full[['Buy', 'Sell']] = df_full[['Buy', 'Sell']].astype(float)
df = df[(df['Date'] > '2021-12-31') & (df['Date'] < '2023-12-31')].reset_index(drop=True)
df[['Buy', 'Sell']] = df[['Buy', 'Sell']].astype(float)

df
#%%
print(len(df), len(df_full))
#%%
# Saving the results of parsing to a file(.csv format)
file_path = Path('data/Eur_Uah.csv')
file_path.parent.mkdir(parents=True, exist_ok=True)

if file_path.is_file():
    print(f'Файл "{file_path}" вже існує.')
else:
    df.to_csv(file_path, index=False)
    print(f'Файл успішно збережено до "{file_path}"')
#%%
def clean_anomalies(df, column='Buy', window=10, sigma=3):
    # Cleaning out anomalies function by mean and std
    df_clean = df.copy()
    rolling_mean = df_clean[column].rolling(window=window).mean()
    rolling_std = df_clean[column].rolling(window=window).std()

    upper_bond = rolling_mean + (sigma * rolling_std)
    lower_bond = rolling_mean - (sigma * rolling_std)

    anomalies = df_clean[(df_clean[column] > upper_bond) | (df_clean[column] < lower_bond)]

    df_clean.loc[anomalies.index, column] = np.nan
    df_clean[column] = df_clean[column].interpolate(method='polynomial', order=5)

    return df_clean, anomalies

def clean_anomalies_by_volatility(df, column='Buy', sigma=3):
    # Cleaning out anomalies function percentage change
    df_clean = df.copy()
    df_clean['returns'] = df_clean[column].pct_change()

    returns_mean = df_clean['returns'].mean()
    returns_std = df_clean['returns'].std()

    upper_threshold = returns_mean + (sigma * returns_std)
    lower_threshold = returns_mean - (sigma * returns_std)

    anomaly_mask = (df_clean['returns'] > upper_threshold) | (df_clean['returns'] < lower_threshold)
    anomalies = df_clean[anomaly_mask]

    df_clean.loc[anomaly_mask, column] = np.nan
    df_clean[column] = df_clean[column].interpolate(method='polynomial', order=5)

    return df_clean.drop(columns=['returns']), anomalies

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

def models_resid_verification(dtf, resid_real, resid_synthetic):
    # Function for models resid verification from previous laboratorium(maybe will be useful in future)
    print(dtf[['Synthetic_Buy', 'Buy']].describe())

    fig, ax = plt.subplots(1, 2, figsize=(18, 5))

    ax[0].hist(resid_real, label="Real", edgecolor='black')
    ax[0].set_xlabel('Resid of the real data')
    ax[0].legend(loc="best")

    ax[1].hist(resid_synthetic, label="Synthetic", edgecolor='black')
    ax[1].set_xlabel('Resid of the synthetic data')
    ax[1].legend(loc="best")
    plt.show()
#%%
# Got anomalies data, but since it's currency price for buy/sell - there will be no major anomalies, which will be the reason to continue laboratorium with my main dataframe - df
df_clean, anomalies_buy = clean_anomalies(df, 'Buy')
df_clean, anomalies_sell = clean_anomalies(df_clean, 'Sell')
draw_plots(df, 'Date', 'Buy')
#%%
print('Anomalies: ')
anomalies_buy[['Date', 'Buy']].join(anomalies_sell[['Sell']], lsuffix='_buy', rsuffix='_sell', how='left')
#%%
df_clean_1, anomalies_buy = clean_anomalies_by_volatility(df, 'Buy')
df_clean_1, anomalies_sell = clean_anomalies_by_volatility(df_clean_1, 'Sell')
#%%
print('Anomalies by volatility: ')
# pd.concat([anomalies_buy.drop('Sell', axis=1), anomalies_sell.drop('Buy', axis=1)])
# pd.concat([anomalies_buy, anomalies_sell])
anomalies_buy[['Date', 'Buy', 'returns']].join(anomalies_sell[['Sell', 'returns']], lsuffix='_buy', rsuffix='_sell', how='left')
#%%
def optimize_polynomial_degree(df, col='Buy', max_degree=10):
    y = df[col]
    x = np.arange(len(df))

    results_stats = []

    print(f"{'Degree':<10} | {'AIC':<15} | {'BIC':<15} | {'R-squared':<15}")

    best_aic = float('inf')
    best_model_results = None

    for deg in range(1, max_degree + 1):
        X_poly = np.vander(x, deg + 1, increasing=True)

        model = sm.OLS(y, X_poly)
        results = model.fit()

        aic = results.aic
        bic = results.bic
        r2 = results.rsquared

        results_stats.append({'degree': deg, 'AIC': aic, 'BIC': bic, 'R2': r2})
        print(f"{deg:<10} | {aic:<15.4f} | {bic:<15.4f} | {r2:<15.4f}")

        if aic < best_aic:
            best_aic = aic
            best_model_results = results

    return pd.DataFrame(results_stats), best_model_results
#%%
# Normality of data
norm_df = df.copy().drop('Sell', axis=1)
norm_df[['Buy']] = StandardScaler().fit_transform(df[['Buy']])

# Getting results for the optimal polynomial degree
stats_df, best_model = optimize_polynomial_degree(norm_df, col='Buy')

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Ступінь полінома')
ax1.set_ylabel('AIC', color=color)
ax1.plot(stats_df['degree'], stats_df['AIC'], marker='o', color=color, label='AIC')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('R-squared', color=color)
ax2.plot(stats_df['degree'], stats_df['R2'], marker='s', linestyle='--', color=color, label='R^2')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Оптимізація моделі')
plt.show()
#%%
# Getting predictions for 100 days ahead
x_future = np.arange(len(norm_df) + 100)
optimal_degree = 5

X_future_poly = np.vander(x_future, optimal_degree + 1, increasing=True)

prediction = best_model.predict(X_future_poly)
best_model.summary()
#%%
prediction
#%%
# Normality of 'full dataset', which contains information about the last days data, so we will be able to compare extrapolation results
norm_full_df = df_full.copy().drop('Sell', axis=1)
norm_full_df[['Buy']] = StandardScaler().fit_transform(norm_full_df[['Buy']])

fig, ax = plt.subplots(nrows=2, figsize=(10, 5))
ax[0].plot(norm_df['Date'], norm_df['Buy'], label='Реальні дані', color='g')
ax[0].plot(norm_df['Date'], prediction[:len(norm_df)], label='OLS екстраполяція', color='r', linestyle='--')
ax[0].set_xlabel('Порівняння реальних даних та OLS тренду')

ax[1].plot(norm_full_df['Date'].iloc[:len(prediction)], norm_full_df['Buy'].iloc[:len(prediction)], label='Реальні дані', color='g')
ax[1].plot(norm_full_df['Date'].iloc[:len(prediction)], prediction, label='OLS екстраполяція', color='r', linestyle='--')
ax[1].set_xlabel('Порівняння реальних даних та OLS екстраполяції')

plt.subplots_adjust(hspace=0.5)
plt.show()
#%%

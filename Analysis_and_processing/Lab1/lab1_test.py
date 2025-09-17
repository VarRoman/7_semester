#%% md
# ## Завдання І, ІІ рівнів складності – максимально 7, 8 балів відповідно.
# ### Розробити програмний скрипт мовою Python що забезпечує аналіз властивостей і характеристик вихідних даних відповідно до етапів:
# 1. Модель генерації випадкової величини за заданим у табл.1 додатку 1 закону
# розподілу;
# 2. Модель зміни (ідеальний тренд) досліджуваного процесу за заданим у табл.1
# додатку 1 законом;
# 3. Адитивна модель статистичної вибірки відповідно до синтезованих в п.1,2 моделей
# випадкової (стохастична) і невипадкової складових. Параметри закону розподілу та закону
# зміни досліджуваного процесу обрати самостійно.
# 4. Визначення статистичних (числових) характеристик сформованих в п.1,3 вибірок
# (дисперсія, середньоквадратичне відхилення, математичне очікування, гістограма закону
# розподілу).
# 5. Визначення статистичних характеристик реальних даних, заданих файлом
# Oschadbank (USD).xls за умов табл. 1 додатку 1.
# 6. Провести аналіз отриманих результатів та верифікацію розробленого скрипта.
# 
# # Завдання ІІІ рівня – максимально 10 балів.
# 1. Провести парсинг самостійно обраного сайту. Вміст даних, що підлягають парсингу
# – обрати самостійно.
# 2. Результати парсингу зберегти у файлі. Тип файлу обрати самостійно.
# 3. Оцінити динаміку тренду реальних даних.
# 4. Здійснити визначення статистичних характеристик результатів парсингу.
# 5. Синтезувати та верифікувати модель даних, аналогічних за трендом і
# статистичними характеристиками реальним даним, які є результатом парсингу.
# 6. Провести аналіз отриманих результатів.
# 
# P.s: *Завдання III рівня було обрано до виконання*
#%% md
# ### 1. Провести парсинг самостійно обраного сайту. Вміст даних, що підлягають парсингу
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
df = df[(df['Date'] < '2024-12-31') & (df['Date'] > '2023-12-31')].reset_index(drop=True)
df[['Buy', 'Sell']] = df[['Buy', 'Sell']].astype(float)

df
#%% md
# ### 2. Результати парсингу зберегти у файлі. Тип файлу обрати самостійно.
#%%
# Saving the results of parsing to a file(.csv format)
file_path = Path('data/Eur_Uah.csv')
file_path.parent.mkdir(parents=True, exist_ok=True)

if file_path.is_file():
    print(f'Файл "{file_path}" вже існує.')
else:
    df.to_csv(file_path, index=False)
    print(f'Файл успішно збережено до "{file_path}"')
#%% md
# ### 3. Оцінити динаміку тренду реальних даних.
# 
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
#%%
draw_plots(df, 'Date', 'Buy')
#%%
# Decomposition
df_decomposed = df.set_index('Date')
decomposition = seas.seasonal_decompose(x=df_decomposed['Buy'], period=5, model='additive')
df_decomposed = pd.concat([df_decomposed, decomposition.trend, decomposition.seasonal, decomposition.resid], axis=1)

residuals = decomposition.resid.dropna()
observed_values = decomposition.observed.loc[residuals.index]
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((observed_values - observed_values.mean()) **2 )

if ss_tot == 0:
    R2_decompose = 1.0
else:
    R2_decompose = 1 - (ss_res / ss_tot)

print(f"Показник R^2 для моделі декомпозиції: {R2_decompose:.4f}")

df_decomposed
#%%
# Setting some additional parameters for the next plots
fig, ax = plt.subplots(2, 2, layout='constrained', width_ratios=[0.6, 0.4])
fig.set_figheight(6)
fig.set_figwidth(10)
ax[0][0].tick_params(axis='x', labelrotation=45)
ax[0][0].set_xlabel('Графік показника сезонності протягом 2024 року')
ax[0][1].set_xticks(ticks=df[::5]['Date'], labels=df[::5]['Date'].astype(str), rotation=45, size=8)
ax[0][1].set_xlabel('Графік показника сезонності протягом січня 2024 ')
ax[1][0].tick_params(axis='x', labelrotation=45)
ax[1][0].set_xlabel('Графік чистого тренду без включення елементів сезонності та \n"шуму"(залишкової компоненти)')
ax[1][1].tick_params(axis='x', labelrotation=45)
ax[1][1].set_xlabel('Графік залишкової компоненти(шуму)')
# fig.tight_layout()

ax[0][0].plot(df_decomposed['seasonal'], color='g')
ax[0][1].plot(df_decomposed['seasonal']['2024/01/01':'2024/02/01'], color='g', )
ax[1][0].plot(df_decomposed['trend'], color='b')
ax[1][1].plot(df_decomposed['resid'], color='b');
#%%
if df_decomposed.index.dtype != 'int64':
    df_decomposed.reset_index(inplace=True)
    df_decomposed.drop(columns=['Sell'], inplace=True)
df_decomposed
#%% md
# ### 4. Здійснити визначення статистичних характеристик результатів парсингу.
#%%
# Normality of data
norm_df = df.copy().drop('Sell', axis=1)
norm_df[['Buy']] = StandardScaler().fit_transform(df[['Buy']])

y = norm_df['Buy']

x = np.arange(len(norm_df))
X_poly = pd.DataFrame({'x': x, 'x^2': x**2, 'x^3': x**3, 'x^4': x**4, 'x^5': x**5})

X_poly_const = sm.add_constant(X_poly)

model = sm.OLS(y, X_poly_const)
results = model.fit()
print(results.summary())

print(f'\nСереднє квадратичне відхилення реальних даних(стовпчика "Buy"): {df['Buy'].std()}')
print(f'Дисперсія реальних даних(стовпчика "Buy"): {df['Buy'].var()}')
print(f'Математичне сподівання(середнє значення стовпчика "Buy" реальних даних): {df['Buy'].mean()}')

print(f'\nСереднє квадратичне відхилення нормалізованих даних(стовпчика "Buy"): {norm_df['Buy'].std()}')
print(f'Дисперсія нормалізованих даних(стовпчика "Buy"): {norm_df['Buy'].var()}')
print(f'Математичне сподівання(середнє значення стовпчика "Buy" нормалізованих даних): {norm_df['Buy'].mean()}')
#%%
norm_df['OLS_trend'] = results.predict(X_poly_const)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(norm_df['Date'], norm_df['Buy'], label='Реальні дані', color='g')
ax.plot(norm_df['Date'], norm_df['OLS_trend'], label='Тренд OLS', color='r', linestyle='--')
ax.set_xlabel('Квадратичний тренд, побудований за допомогою OLS')
ax.plot();

#%% md
# ### 5. Синтезувати та верифікувати модель даних, аналогічних за трендом і статистичними характеристиками реальним даним, які є результатом парсингу.
# *Ця частина буде складатися з 2 етапів, бо алгоритмів, які ми будемо тут розглядати теж 2*
#%%
# Function for showing the trend and borders of resid
def trend_resid_show(dtf, col_resid_name, col_trend_name, model_name, axi):
    dtf[col_resid_name] = dtf['Buy'] - dtf[col_trend_name]
    model_resid_std = dtf[col_resid_name].std()
    dtf['upper_3sigma'] = dtf[col_trend_name] + 3 * model_resid_std
    dtf['lower_3sigma'] = dtf[col_trend_name] - 3 * model_resid_std
    print('Аномальні дані:')
    print(dtf[dtf[col_resid_name] > 3 * model_resid_std])

    # Deleting anomalies
    # dtf['Buy'] = dtf['Buy'].mask(dtf[col_resid_name] > 3 * model_resid_std, dtf[col_trend_name], axis=0)  # this line is for changing to the trend value
    dtf.drop(index=list(dtf[dtf[col_resid_name] > 3 * model_resid_std].index), inplace=True)

    # fig, ax = plt.subplots(figsize=(10, 5))

    axi.plot(dtf['Date'], dtf[col_trend_name], "r", label=model_name)
    axi.plot(dtf['Date'], dtf['Buy'], "b-", label="Real")
    axi.plot(dtf['Date'], dtf['upper_3sigma'], "r--")
    axi.plot(dtf['Date'], dtf['lower_3sigma'], "r--")
    axi.set_xlabel('Trend with real data')
    axi.legend(loc="best")

def gen_synthetic_data(dtf, col_resid_name, col_trend_name, model_name):
    # Synthesising synthetic data and applying it to the trend values
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    trend_resid_show(dtf, col_resid_name, col_trend_name, model_name, ax[0])

    model_resid_std = dtf[col_resid_name].std()
    model_synthetic_noise = np.random.normal(loc=0, scale=model_resid_std, size=len(dtf))
    dtf['Synthetic_Buy'] = dtf[col_trend_name] + model_synthetic_noise

    ax[1].plot(dtf['Date'], dtf[col_trend_name], "r", label=model_name)
    ax[1].plot(dtf['Date'], dtf['Synthetic_Buy'], "y", label=f"Synthetic {model_name} model")
    ax[1].plot(dtf['Date'], dtf['upper_3sigma'], "r--")
    ax[1].plot(dtf['Date'], dtf['lower_3sigma'], "r--")
    ax[1].set_xlabel('Trend with synthetic data')
    ax[1].legend(loc="best")
    plt.show()
#%% md
# 1. Отриманий за допомогою МНК та моделі OLS
#%%
# First model
# trend_resid_show(norm_df, 'OLS_resid', 'OLS_trend', 'OLS')
gen_synthetic_data(norm_df, 'OLS_resid', 'OLS_trend', 'OLS')
#%% md
# 2. Отриманий під час декомпозиції
#%%
# Second model
# trend_resid_show(df_decomposed, 'resid', 'trend', 'Seasonal')
gen_synthetic_data(df_decomposed, 'resid', 'trend', 'Seasonal')
#%%
norm_df
#%%
df_decomposed
#%% md
# ### Верифікація моделей
#%%
def models_resid_verification(dtf, resid_real, resid_synthetic):
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
print('OLS model')
models_resid_verification(norm_df, norm_df['OLS_resid'], norm_df['Synthetic_Buy'] - norm_df['OLS_trend'])
#%%
print('Seasonal decomposition model')
models_resid_verification(df_decomposed, df_decomposed['resid'], df_decomposed['Synthetic_Buy'] - df_decomposed['trend'])
#%%

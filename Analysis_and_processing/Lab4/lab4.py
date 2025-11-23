#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.python.ops.ragged.ragged_array_ops import cross_hashed
#%%
df = pd.read_excel('Data_Set/Data_Set_7.xlsx', sheet_name='qrySales')
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

pivot_df = df.pivot_table(index='OrderDate', columns='CustomerCountry', values='Revenue', aggfunc='sum').fillna(0)

monthly_sales = pivot_df.resample('ME').sum()
monthly_sales.head()
#%%
df
#%%
def calculate_ols_slope(series):
    y = series.values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return slope

class AlphaBetaGammaFilter:
    def __init__(self, alpha=0.25, beta=0.005, gamma=0.001,
                 initial_value=None, initial_velocity=0, initial_accel=0,
                 adaptive=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.x = initial_value
        self.v = initial_velocity
        self.a = initial_accel

        self.adaptive = adaptive
        self.estimates = []

    def my_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update(self, measurement, dt=1):
        if self.x is None:
            self.x = measurement
            self.estimates.append(self.x)
            return self.x

        x_pred = self.x + self.v * dt + 0.5 * self.a * (dt ** 2)
        v_pred = self.v + self.a * dt
        a_pred = self.a

        residual = measurement - x_pred

        if self.adaptive:
            min_alpha = 0.1
            max_alpha = 0.5
            center = 0.5
            slope = 2

            sigmoid_input = (abs(residual) - center) * slope
            self.alpha = min_alpha + (max_alpha - min_alpha) * self.my_sigmoid(sigmoid_input)
            self.beta = 2 * (2 - self.alpha) - 4 * np.sqrt(1 - self.alpha)
            self.gamma = (self.beta ** 2) / (2 - self.alpha)

        self.x = x_pred + self.alpha * residual
        self.v = v_pred + self.beta / dt * residual
        self.a = a_pred + (2 * self.gamma) / (dt ** 2) * residual

        self.estimates.append(self.x)
        return self.x

    def extrapolate(self, steps, dt=1):
        extrapol_results = []
        temp_x = self.x
        temp_v = self.v
        temp_a = self.a

        for i in range(steps):
            temp_x = temp_x + temp_v * dt + 0.5 * temp_a * (dt ** 2)
            temp_v = temp_v + temp_a * dt
            extrapol_results.append(temp_x)

        return np.array(extrapol_results)

def apply_kalman_smoothing(series):
    filter_1 = AlphaBetaGammaFilter(initial_value=series.iloc[0])
    smoothed_values = []

    for val in series:
        smoothed_values.append(filter_1.update(val))
    return pd.Series(smoothed_values, index=series.index)
#%%
smoothed_sales = monthly_sales.copy()

for country in monthly_sales.columns:
    smoothed_sales[country] = apply_kalman_smoothing(monthly_sales[country])

random_sample = monthly_sales.sum().idxmax()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales[random_sample], label='Оригінальні дані', alpha=0.5, color='gray')
plt.plot(smoothed_sales[random_sample], label='Після фільтра Калмана', color='blue')
plt.title(f'Ефект згладжування для {random_sample}')
plt.legend()
plt.show()
#%%
target_series = smoothed_sales[random_sample]

decomposition = seasonal_decompose(target_series, model='additive', period=6)

fig = decomposition.plot()
fig.set_size_inches(8, 6)
plt.suptitle(f"Декомпозиція часового ряду:", fontsize=16, y=1)

plt.show()
#%%
smoothed_sales
#%%
corr_matrix = smoothed_sales.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, vmin=-1, vmax=1)
plt.title("Кореляція продажів між країнами", fontsize=16)
plt.show()

corr_pairs = corr_matrix.unstack(fill_value='hey').sort_values(ascending=False)
corr_pairs = corr_pairs[(corr_pairs < 0.999)]

print("ТОП-5 пар країн з найбільш синхронною динамікою продажів:")
print(corr_pairs.head(10)[::2])

print("\nТОП-3 пари з оберненою кореляцією:")
print(corr_pairs.tail(6)[::2])
#%%
features = pd.DataFrame(index=monthly_sales.columns)
features['Mean_Sales'] = monthly_sales.mean()
features['Trend_Slope'] = smoothed_sales.apply(calculate_ols_slope)
features['Volatility'] = monthly_sales.std()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[['Mean_Sales', 'Trend_Slope', 'Volatility']])

k_range = range(2, 10)
best_score = -1
best_k = -1
scores = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(features_scaled)

    score = silhouette_score(features_scaled, labels)
    scores.append(score)

    print(f'Для k={k}, Silhouette Score = {score:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(k_range, scores, marker='o')
plt.title('Коефіцієнт Силуету для визначення оптимального k')
plt.xlabel('Кількість кластерів')
plt.ylabel('Значення коефіцієнту Силуету')
plt.show()
#%%
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
features['Cluster'] = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=features, x='Mean_Sales', y='Trend_Slope',
                hue='Cluster', palette='viridis', s=100, style='Cluster')

for i in range(features.shape[0]):
    if features.Mean_Sales.iloc[i] > features.Mean_Sales.mean():
        plt.text(features.Mean_Sales.iloc[i], features.Trend_Slope.iloc[i],
                 features.index[i], fontsize=9)

plt.title("Кластеризація ринків збуту", fontsize=16)
plt.xlabel("Середній обсяг продажів")
plt.ylabel("Тренд розвитку")
plt.show()
#%%
cluster_stats = features.groupby('Cluster')[['Mean_Sales', 'Trend_Slope', 'Volatility']].mean()
print("Середні показники по кластерах:")
display(cluster_stats)

plt.figure(figsize=(10, 4))
for cluster_id in range(n_clusters):
    countries_in_cluster = features[features['Cluster'] == cluster_id].index
    mean_series = smoothed_sales[countries_in_cluster].mean(axis=1)
    plt.plot(mean_series, label=f'Кластер {cluster_id} (Кількість країн: {len(countries_in_cluster)})')

plt.title("Усереднені патерни поведінки для кожного кластера", fontsize=16)
plt.xlabel("Час")
plt.ylabel("Середні продажі")
plt.legend()
plt.show()
#%%

import numpy as np
import pandas as pd
from pyclustering.cluster.cure import cure
from pyclustering.utils import distance_metric, type_metric
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from data_generator_functions import generate_dataset
from anonymizer_functions import *


def visualize_clusters(data, clusters, title, use_3d=False):
    """Улучшенная визуализация кластеров"""
    if hasattr(data, 'toarray'):
        data = data.toarray()

    pca = PCA(n_components=3 if use_3d else 2)
    data_transformed = pca.fit_transform(data)

    plt.figure(figsize=(12, 8 if use_3d else 10))

    if use_3d:
        ax = plt.subplot(111, projection='3d')
        for i, cluster in enumerate(clusters):
            points = data_transformed[cluster]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=50, label=f'Cluster {i + 1}', alpha=0.7)
        ax.set_zlabel('PC3')
    else:
        ax = plt.gca()
        for i, cluster in enumerate(clusters):
            points = data_transformed[cluster]
            ax.scatter(points[:, 0], points[:, 1],
                       s=50, label=f'Cluster {i + 1}', alpha=0.7)

    plt.title(f'{title}\nExplained variance: {sum(pca.explained_variance_ratio_):.2f}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_distributions(original, anonymized, feature_name):
    """Сравнение распределений признака до и после обезличивания"""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(original, bins=20, alpha=0.7, color='blue')
    plt.title(f'Original {feature_name}')

    plt.subplot(1, 2, 2)
    plt.hist(anonymized, bins=20, alpha=0.7, color='red')
    plt.title(f'Anonymized {feature_name}')

    plt.tight_layout()
    plt.show()


def prepare_data(df, is_anonymized=False):
    df_processed = df.copy()
    columns_to_drop = ['ФИО', 'Email', 'Телефон', 'Паспортные данные', 'Карта оплаты']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])

    if is_anonymized:
        age_mapping = {
            '<18': 16,
            '18-24': 21,
            '25-34': 29,
            '35-49': 42,
            '50-64': 57,
            '65+': 70
        }
        if 'Возраст' in df_processed.columns:
            df_processed['Возраст'] = df_processed['Возраст'].map(age_mapping)

        price_mapping = {
            '< 1000': 500,
            '1000 - 2000': 1500,
            '2000 - 4000': 3000,
            '4000 - 5000': 4500,
            '>=5000': 5500
        }
        if 'Стоимость' in df_processed.columns:
            df_processed['Стоимость'] = df_processed['Стоимость'].map(price_mapping)

        df_processed = df_processed.drop(columns=['Дата отъезда', 'Дата приезда'], errors='ignore')
    else:
        for date_col in ['Дата отъезда', 'Дата приезда']:
            if date_col in df_processed.columns:
                df_processed[date_col] = pd.to_datetime(
                    df_processed[date_col], format='%Y-%m-%dT%H:%M', errors='coerce'
                ).astype('int64') // 10 ** 9

    numeric_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    processed_data = preprocessor.fit_transform(df_processed)
    if hasattr(processed_data, 'toarray'):
        processed_data = processed_data.toarray()

    return processed_data


def calculate_compactness(data, clusters):
    """Метрика компактности с квадратом евклидова расстояния"""
    total = 0.0
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        points = data[cluster]
        centroid = np.mean(points, axis=0)
        distances = pairwise_distances(points, [centroid], metric='sqeuclidean')
        total += np.sum(distances)
    return total / len(clusters) if clusters else 0.0


def run_cure(data, n_clusters=5):
    metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)
    cure_instance = cure(
        data=data.tolist(),
        number_cluster=n_clusters,
        number_represent_points=5,
        compression=0.2
    )
    cure_instance.process()
    return cure_instance.get_clusters()


def select_features(data, n_features=5, n_jobs=-1):
    def process_feature(i):
        clusters = run_cure(data[:, [i]], n_clusters=3)
        return (i, calculate_compactness(data[:, [i]], clusters))

    scores = Parallel(n_jobs=n_jobs)(delayed(process_feature)(i) for i in range(data.shape[1]))
    scores.sort(key=lambda x: x[1])
    return [x[0] for x in scores[:n_features]]


# Основной процесс
# 1. Генерация данных
print("Генерация датасета...")
generate_dataset(banks_prob=[40, 30, 20, 10], payment_systems_prob=[50, 30, 20], n_rows=1000)
df = pd.read_csv('original_dataset.csv')

# 2. Обезличивание данных
print("Обезличивание данных...")


def anonymize_data(df):
    anonymized = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        de_name(row_dict)
        de_age(row_dict)
        de_email(row_dict)
        de_phone(row_dict)
        de_passport(row_dict)
        de_marsh(row_dict)
        de_train_type(row_dict)
        de_wagon(row_dict)
        de_price(row_dict)
        de_card(row_dict)
        de_date(row_dict)
        anonymized.append(row_dict)
    return pd.DataFrame(anonymized)


anonymized_df = anonymize_data(df)
anonymized_df.to_csv('anonymized_dataset.csv', index=False)

# 3. Визуализация распределений
print("Визуализация распределений...")
compare_distributions(df['Возраст'], anonymized_df['Возраст'], 'Age')
compare_distributions(df['Стоимость'], anonymized_df['Стоимость'], 'Price')

# 4. Подготовка данных
print("Подготовка данных...")
X_original = prepare_data(df, is_anonymized=False)
X_anonymized = prepare_data(anonymized_df, is_anonymized=True)

# 5. Эксперименты
results = {}
phases = [
    ('Исходные данные', X_original, False),
    ('После отбора признаков (исходные)', X_original, True),
    ('Обезличенные данные', X_anonymized, False),
    ('После отбора признаков (обезличенные)', X_anonymized, True)
]

for name, data, use_feature_selection in phases:
    print(f"\nОбработка: {name}")

    if use_feature_selection:
        selected_cols = select_features(data)
        data = data[:, selected_cols]

    clusters = run_cure(data)
    compactness = calculate_compactness(data, clusters)
    results[name] = compactness

    # Визуализация
    visualize_clusters(data, clusters, name, use_3d=True)
    visualize_clusters(data, clusters, name)

# 6. Вывод результатов
print("\n=== Итоговые результаты ===")
for name, value in results.items():
    print(f"{name}: {value:.2f}")

# Анализ
print("\n=== Анализ ===")
print("1. Сравнение исходных данных и обезличенных:")
print(f"   Изменение компактности: {results['Обезличенные данные'] - results['Исходные данные']:.2f}")

print("\n2. Эффект отбора признаков:")
print(f"   Для исходных данных: {results['Исходные данные']} → {results['После отбора признаков (исходные)']}")
print(f"   Для обезличенных: {results['Обезличенные данные']} → {results['После отбора признаков (обезличенные)']}")
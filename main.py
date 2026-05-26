#import matplotlib.pyplot as plt

#fig,ax=plt.subplots()
#ax.plot([1,2,3,4],[1,4,2,5])
#plt.ylabel('some numbers')
#plt.savefig('myplot.png')

# ============================================
# БЛОК 1: ГЕНЕРАЦИЯ И ЗАГРУЗКА ДАННЫХ
# ============================================
from sklearn.datasets import make_blobs
import pandas as pd

# Генерация синтетического набора данных: 200 точек, 2 признака, 4 центра
dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)

# Преобразование в DataFrame для удобной работы
df = pd.DataFrame(dataset, columns=['var1', 'var2'])

# Вывод первых двух строк для контроля (аналог "вывести первые строки" из блок-схемы)
print(df.head(2))


# ============================================
# БЛОК 2: ВЫБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ (МЕТОД ЛОКТЯ)
# ============================================
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Расчет инерции (суммы квадратов ошибок) для k от 1 до 11
# Используется List Comprehension для компактности
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(df).inertia_ for k in range(1, 12)]

# Построение графика зависимости инерции от количества кластеров
plt.plot(range(1, 12), inertias, 'bo-')  # 'bo-' — синие кружки с линией
plt.grid(True)  # Добавление сетки для наглядности
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.savefig('elbow.png')  # Сохранение графика в файл
plt.close()


# ============================================
# БЛОК 3: ФИНАЛЬНАЯ КЛАСТЕРИЗАЦИЯ С ОПТИМАЛЬНЫМ k
# ============================================
# Примечание: В вашей блок-схеме оптимальное k выбирается по излому графика.
# В коде вы жестко задали k=4, но по логике блок-схемы здесь должно быть:
# optimal_k = find_elbow(inertias)  # Анализ графика
# kmeans = KMeans(n_clusters=optimal_k, ...)

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)

# Извлечение результатов кластеризации
print(kmeans.labels_)          # Метки кластеров для каждой точки
print(kmeans.cluster_centers_) # Координаты центров кластеров
print(kmeans.inertia_)         # Сумма квадратов ошибок
print(kmeans.n_iter_)          # Количество итераций


# ============================================
# БЛОК 4: АНАЛИЗ РАСПРЕДЕЛЕНИЯ ТОЧЕК ПО КЛАСТЕРАМ
# ============================================
from collections import Counter

# Подсчет количества точек в каждом кластере (аналог "размер каждого кластера" из схемы)
print(Counter(kmeans.labels_))


# ============================================
# БЛОК 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (ДИАГРАММА РАССЕИВАНИЯ)
# ============================================
import seaborn as sns

# Построение диаграммы рассеивания с цветовой кодировкой по кластерам
sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)

# Добавление центров кластеров на график (красные крестики)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker="X", c="r", s=80, label="Centroids")

plt.legend()  # Отображение легенды
plt.grid(True, alpha=0.3)  # Сетка с прозрачностью 0.3
plt.xlabel('var1')
plt.ylabel('var2')
plt.title('Результаты кластеризации')
plt.savefig('neww.png')  # Сохранение итогового графика
plt.close()


# ==================== ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ====================
# ================================================================
# БЛОК 1: ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
# ================================================================
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# ================================================================
# БЛОК 2: ГЕНЕРАЦИЯ ДАННЫХ (соответствует блоку "Генерация набора данных")
# ================================================================
# Параметры для генерации
n_samples = 200
n_features = 2
centers = 3

# Генерируем числовые данные (синтетический набор с 3 центрами)
dataset, numeric_classes = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=centers,
    cluster_std=0.5,
    random_state=0
)

# Масштабируем параметры в нужные диапазоны (0.01-1 и 1-300)
# Ключевая строка: интерполяция данных в заданные границы
param1 = np.interp(dataset[:, 0], [dataset[:, 0].min(), dataset[:, 0].max()], [0.01, 1])
param2 = np.interp(dataset[:, 1], [dataset[:, 1].min(), dataset[:, 1].max()], [1, 300])

# Генерируем параметр 3 (города) - соответствие классам make_blobs
city_mapping = {0: 'Самара', 1: 'Тольятти', 2: 'Москва'}
param3 = [city_mapping[cls] for cls in numeric_classes]

# Создаем DataFrame
df = pd.DataFrame({
    'param1': param1,
    'param2': param2,
    'city': param3
})

# Вывод первых строк (соответствует блоку "Вывести в консоль")
print("Сгенерированные данные (первые 10 записей):")
print(df.head(10))
print(f"\nРазмер данных: {df.shape}")
print(f"\nРаспределение по городам:\n{Counter(df['city'])}")

# ================================================================
# БЛОК 3: ПРЕОБРАЗОВАНИЕ ТЕКСТОВЫХ ЗНАЧЕНИЙ В ЧИСЛОВОЙ КОД
# (соответствует блоку "Преобразование текстовых значений...")
# ================================================================
print("\n" + "="*50)
print("ПУНКТ 5: ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
print("="*50)

# Кодируем города с помощью LabelEncoder
label_encoder = LabelEncoder()
df['city_encoded'] = label_encoder.fit_transform(df['city'])
print("Кодирование городов:")
for city, code in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"  {city} -> {code}")

# ================================================================
# БЛОК 4: НОРМИРОВАНИЕ ЧИСЛОВЫХ ПАРАМЕТРОВ
# (соответствует блоку "Нормирование числовых параметров")
# ================================================================
# Матрица признаков (только числовые параметры для визуализации)
X = df[['param1', 'param2']].values

# Стандартизация (нормирование) признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================
# БЛОК 5: ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ (МЕТОД ЛОКТЯ)
# (соответствует блоку "Определение оптимального количества кластеров...")
# ================================================================
print("\n" + "="*50)
print("ПУНКТ 6: КЛАСТЕРИЗАЦИЯ И ВИЗУАЛИЗАЦИЯ")
print("="*50)

# Расчет инерции для k от 1 до 11
print("\nОпределение оптимального количества кластеров...")
# Ключевая строка: List Comprehension для расчета инерции
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled).inertia_ for k in range(1,12)]

# Построение графика метода локтя
plt.plot(range(1,12), inertias, 'bo-')
plt.grid(True)
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.savefig('elbow_method_kmeans.png')
plt.close()
print("График 'elbow_method_kmeans.png' сохранен")

# ================================================================
# БЛОК 6: ВЫПОЛНЕНИЕ КЛАСТЕРИЗАЦИИ С ОПТИМАЛЬНЫМ k
# (соответствует блоку "Выполнение кластеризации...")
# ================================================================
# Выбор оптимального k (в данном примере - по графику локтя выбрано 3)
optimal_k = 3
print(f"\nВыполнение кластеризации с k={optimal_k}...")

# Ключевая строка: инициализация и обучение модели KMeans
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0, n_init=10)
kmeans.fit(X_scaled)
df['cluster'] = kmeans.labels_

# Обратное масштабирование центроидов в исходные координаты
# Ключевая строка: обратное преобразование для визуализации
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

# ================================================================
# БЛОК 7: ВЫВОД РЕЗУЛЬТАТОВ В КОНСОЛЬ
# (соответствует блоку "Вывести в консоль: распределение точек...")
# ================================================================
print("\nРезультаты кластеризации:")
print(f"Метки кластеров: {Counter(kmeans.labels_)}")
print(f"Инерция: {kmeans.inertia_}")
print(f"Количество итераций: {kmeans.n_iter_}")
print(f"\nЦентроиды в исходных координатах (param1, param2):")
for i, centroid in enumerate(centroids_original):
    print(f"  Кластер {i}: param1={centroid[0]:.3f}, param2={centroid[1]:.1f}")

# Сравнение реальных городов с кластерами (анализ соответствия)
print("\nСравнение реальных городов с кластерами:")
print(pd.crosstab(df['city'], df['cluster']))

# ================================================================
# БЛОК 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (ДИАГРАММА РАССЕИВАНИЯ)
# (соответствует блоку "Построить диаграмму рассеивания и сохранить")
# ================================================================
# Построение итоговой диаграммы рассеивания
plt.figure()
plt.scatter(df['param1'], df['param2'], c=df['cluster'], cmap='viridis', s=50)
# Добавление центроидов на график (используются обратно масштабированные координаты)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            marker="X", c="r", s=200, label="Centroids")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel('param1')
plt.ylabel('param2')
plt.title('Результаты кластеризации')
plt.savefig('clustering_results_kmeans.png')
plt.close()
print("График 'clustering_results_kmeans.png' сохранен")

print("\n" + "="*50)
print("МИКРОСЕРВИС УСПЕШНО ВЫПОЛНИЛ ЗАДАЧУ")
print("="*50)
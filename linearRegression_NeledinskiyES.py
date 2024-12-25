import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Данные
data = {
    'Год': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Уровень занятости': [55.5, 54.5, 54.5, 52.9, 53.4, 53.8, 55.0]
}

# Создание DataFrame
df = pd.DataFrame(data)

# Подготовка данных для модели
X = df[['Год']]  # Признак (независимая переменная)
y = df['Уровень занятости']  # Целевая переменная

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Прогнозирование уровня занятости на следующий год (2024)
next_year = np.array([[2024]])
predicted_employment_rate = model.predict(next_year)

print(f"Прогноз уровня занятости на 2024 год: {predicted_employment_rate[0]:.2f}%")

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.scatter(df['Год'], df['Уровень занятости'], color='blue', label='Фактические данные')
plt.plot(df['Год'], model.predict(X), color='red', label='Линейная регрессия')
plt.scatter(2024, predicted_employment_rate, color='green', label='Прогноз на 2024', s=100)
plt.title('Уровень занятости населения республики Карелия')
plt.xlabel('Год')
plt.ylabel('Уровень занятости (%)')
plt.xticks(df['Год'].tolist() + [2024])  # Добавляем 2024 в метки по оси X
plt.legend()
plt.grid()
plt.show()

import pandas as pd
import numpy as np
import random
import re
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as MSE
from joblib import dump

# Устанавливаем случайные семена для воспроизводимости результатов
random.seed(42)
np.random.seed(42)

# Загружаем данные
df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

# Удаляем дубликаты строк
df_train = df_train.drop_duplicates()

# Создаем маску для столбцов, исключая цену продажи
mask = ~df_train.columns.isin(['selling_price'])
columns_to_check = df_train.columns[mask]

# Оставляем только уникальные строки по указанным столбцам
df_train = df_train.drop_duplicates(subset=columns_to_check, keep='first').reset_index(drop=True)

# Функция для извлечения чисел из строковых значений
def extract_number_with_regex(value):
    if pd.isnull(value):
        return np.nan
    value_str = str(value)
    match = re.search(r'\d+\.?\d*', value_str)
    return float(match.group()) if match else np.nan

# Применяем функцию к столбцам с числовыми значениями
for col in ['mileage', 'engine', 'max_power']:
    df_train[col] = df_train[col].apply(lambda x: extract_number_with_regex(x))
    df_test[col] = df_test[col].apply(lambda x: extract_number_with_regex(x))

# Удаление столбца torque
df_train = df_train.drop('torque', axis=1)
df_test = df_test.drop('torque', axis=1)

# Список столбцов с пропусками данных
columns_with_gaps = ['mileage', 'engine', 'max_power', 'seats']

# Заполняем пропуски медианными значениями
for column in columns_with_gaps:
    df_train[column] = df_train[column].fillna(df_train[column].median())
    df_test[column] = df_test[column].fillna(df_train[column].median())

# Приводим столбцы engine и seats к целочисленному типу
int_columns = ['engine', 'seats']

for column in int_columns:
    df_train[column] = df_train[column].astype(int)
    df_test[column] = df_test[column].astype(int)

# Разделение столбцов на числовые и категориальные
numerical_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

# Работа с категориальными признаками
X_train_cat = df_train.drop(columns=['name', 'selling_price'], axis=1)
X_test_cat = df_test.drop(columns=['name', 'selling_price'], axis=1)
y_train = df_train['selling_price']
y_test = df_test['selling_price']

numerical_columns = X_train_cat.columns.difference(categorical_cols + ['seats'])

# Преобразование категориальных признаков
encoder = OneHotEncoder(drop='first').fit(X_train_cat[categorical_cols])

train_encoded = encoder.transform(X_train_cat[categorical_cols])

category_column_names = encoder.get_feature_names_out().tolist()

train_encoded_df = pd.DataFrame(train_encoded.toarray(), columns=category_column_names)

X_train_final = pd.concat([X_train_cat[numerical_columns], X_train_cat['seats'], train_encoded_df], axis=1)

test_encoded = encoder.transform(X_test_cat[categorical_cols])

test_encoded_df = pd.DataFrame(test_encoded.toarray(), columns=category_column_names)

X_test_final = pd.concat([X_test_cat[numerical_columns], X_test_cat['seats'], test_encoded_df], axis=1)

# Оценка модели на тестовом наборе

ridge_model = Ridge()

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(Ridge(), param_grid, cv=10, scoring='r2', n_jobs=-1)

grid_search.fit(X_train_final, y_train)

best_alpha = grid_search.best_params_['alpha']

ridge_best = grid_search.best_estimator_

y_pred_train = ridge_best.predict(X_train_final)
y_pred_test = ridge_best.predict(X_test_final)

r2_train = r2_score(y_train, y_pred_train)
mse_train = MSE(y_train, y_pred_train)

r2_test = r2_score(y_test, y_pred_test)
mse_test = MSE(y_test, y_pred_test)

print(f"Оптимальное значение alpha: {best_alpha}")
print(f"R2 для тренировочных данных: {r2_train:.3f}")
print(f"MSE для тренировочных данных: {mse_train:.3f}\n")

print(f"R2 для тестовых данных: {r2_test:.3f}")
print(f"MSE для тестовых данных: {mse_test:.3f}\n")

# Сохранение лучшей модели в файл
dump(ridge_best, 'best_model.joblib')
dump(encoder, 'encoder.joblib')
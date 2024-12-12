import io
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from joblib import load
from fastapi.responses import StreamingResponse

app = FastAPI()

# Загружаем модель и другие необходимые элементы
ridge_model = load('best_model.joblib')  # Загружаем сохраненную модель
encoder = load('encoder.joblib')
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

# Описание структуры входных данных через Pydantic
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: int

def preprocess_input(input_data, encoder, categorical_cols):
    """
    Обрабатывает входные данные для подачи в модель.
    """
    # Приведение к формату DataFrame
    df = pd.DataFrame(input_data)
    # Извлечение чисел из строковых данных
    def extract_number_with_regex(value):
        value_str = str(value)
        match = re.search(r'\d+\.?\d*', value_str)
        return float(match.group()) if match else np.nan

    # Обработка числовых колонок
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(lambda x: extract_number_with_regex(x))

    # Удаление столба torque
    df = df.drop(columns=['name', 'torque'], axis=1)

    # Приведение типов
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)

    # Проверка категорий для OneHotEncoder
    encoder_categories = encoder.categories_
    for i, col in enumerate(categorical_cols):
        df[col] = pd.Categorical(df[col], categories=encoder_categories[i])

    # Обработка категориальных признаков
    numerical_columns = df.columns.difference(categorical_cols + ['seats'])
    encoded = encoder.transform(df[categorical_cols])
    category_column_names = encoder.get_feature_names_out().tolist()
    encoded_df = pd.DataFrame(encoded.toarray(), columns=category_column_names)

    # Финальный набор данных
    df_final = pd.concat([df[numerical_columns], df['seats'], encoded_df], axis=1)

    return pd.DataFrame(df_final)

# Метод для предсказания одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Преобразование объекта в DataFrame
    input_data = item.model_dump()
    data_processed = preprocess_input([input_data], encoder, categorical_cols)

    # Предсказание с использованием модели
    prediction = ridge_model.predict(data_processed)[0]
    return float(prediction)

# Метод для обработки CSV-файла с признаками
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    # Считывание содержимого файла
    contents = await file.read()

    # Преобразование содержимого в DataFrame
    df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')

    # Подготовка данных для предсказаний
    data_processed = preprocess_input(df, encoder, categorical_cols)

    # Получение предсказаний
    predictions = ridge_model.predict(data_processed)

    # Добавление столбца с предсказаниями в исходную таблицу
    df['predicted_selling_price'] = predictions

    # Преобразуем DataFrame обратно в CSV
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Возвращаем CSV-файл
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment;filename=predicted_{file.filename}"
        },
    )
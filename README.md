# AI_HW1_Regression_with_inference_base
Homeork for ML
В результате данной работы удалось сотворить сервис, 
![image](https://github.com/user-attachments/assets/64a8b7f3-8e91-482f-adaa-dd487fd2ae21)
который на вход получает данные в формате json (для примера), --->
{
  "name": "Maruti Wagon R LXI",
  "year": 2010,
  "km_driven": 40000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner",
  "mileage": "15.0 kmpl",
  "engine": "1496 CC",
  "max_power": "100 bhp",
  "torque": "136 Nm",
  "seats": 5
}

а на выходе мы получаем предсказанную цену:
![image](https://github.com/user-attachments/assets/7859961f-4bd1-4d4f-8659-ebdeb4678072)
Кроме этого, сервис способен на вход получать csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах (тестовый файл):
![image](https://github.com/user-attachments/assets/a9b2e6d1-bbff-4eb3-a61c-341352cd409c)
Общий вид запросов выглядит как:
![image](https://github.com/user-attachments/assets/c22e06a0-5036-47b3-a91b-3fc183388892)
Для корректной демонстрации необходимо создать репозитории (или скачать), добавить файлы main.py и app.py, запустить main.py (будут созданы encoder.joblib и best_model.joblib)
В командной строке PyCharm или VSCode запустить сервис командой: uvicorn app:app --host 127.0.0.1 --port 8000
Перейти во вкладку http://127.0.0.1:8000/docs#/.

Точность даннои модели:
Оптимальное значение alpha: 10
R2 для тренировочных данных: 0.636
MSE для тренировочных данных: 104329754899.679
R2 для тестовых данных: 0.639
MSE для тестовых данных: 207550220430.898

Для улучшения работы сервиса необходимо закодировать категориальную переменную "name".

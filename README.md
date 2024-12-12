# AI_HW1_Regression_with_inference_base
Homeork for ML

В результате данной работы удалось разработать сервис, который на вход получает данные в формате json (для примера), --->
{<br/>
  "name": "Maruti Wagon R LXI",<br/>
  "year": 2010,<br/>
  "km_driven": 40000,<br/>
  "fuel": "Petrol",<br/>
  "seller_type": "Individual",<br/>
  "transmission": "Manual",<br/>
  "owner": "First Owner",<br/>
  "mileage": "15.0 kmpl",<br/>
  "engine": "1496 CC",<br/>
  "max_power": "100 bhp",<br/>
  "torque": "136 Nm",<br/>
  "seats": 5<br/>
}<br/>
![image](https://github.com/user-attachments/assets/c22e06a0-5036-47b3-a91b-3fc183388892)

На выходе мы получаем предсказанную цену:
![image](https://github.com/user-attachments/assets/7859961f-4bd1-4d4f-8659-ebdeb4678072)

Кроме этого, сервис способен на вход получать csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах ([тестовый файл](https://github.com/sidorov-ka/AI_HW1_Regression_with_inference_base/blob/master/cars_without_selling_price.csv)):
![image](https://github.com/user-attachments/assets/a9b2e6d1-bbff-4eb3-a61c-341352cd409c)

Для корректной демонстрации необходимо создать репозитории (или скачать), добавить файлы main.py и app.py, запустить main.py (будут созданы encoder.joblib и best_model.joblib)
![image](https://github.com/user-attachments/assets/64a8b7f3-8e91-482f-adaa-dd487fd2ae21)
В командной строке PyCharm или VSCode запустить сервис командой: uvicorn app:app --host 127.0.0.1 --port 8000
Перейти во вкладку http://127.0.0.1:8000/docs#/.

Точность даннои модели:
Оптимальное значение alpha: 10
R2 для тренировочных данных: 0.636
MSE для тренировочных данных: 104329754899.679
R2 для тестовых данных: 0.639
MSE для тестовых данных: 207550220430.898

Для улучшения работы сервиса необходимо закодировать категориальную переменную "name".

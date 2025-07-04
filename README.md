# Отчёт команды «Покорители неба»
Разработка и реализация системы автономной навигации БПЛА с распознаванием букв армянского алфавита и посадкой по маркерам ArUco

##  1. Цель проекта
  Целью проекта являлось создание интеллектуальной системы управления БПЛА, способной:

•	    Распознавать символы армянского алфавита на поверхности при полёте на высоте 30–50 метров; <br>
•	    Локализовать ArUco-маркеры и производить точное центрирование над ними; <br>
•	    Автоматически инициировать посадку при обнаружении маркера;<br>
•	    Работать в режиме реального времени, контролируясь RC-пультом;<br>
•	    Функционировать автономно с возможностью логирования и удалённого мониторинга.<br>
<br>
<br>
## 2. Подготовка датасета<br>
  
###  2.1. Генерация на основе шрифтов<br>

На начальном этапе для каждой буквы армянского алфавита (кодировки Unicode U+0531–U+0556) были сгенерированы изображения на основе шрифтов. Алгоритм генерации предусматривал:<br>

•	Использование шрифтов с поддержкой армянского письма (Mshtakan и другие);<br>
•	Белый фон, чёрная буква по центру;<br>
•	Повороты от –15° до +15°;<br>
•	Добавление размытия (имитация вибрации дрона);<br>
•	Добавление нормального шума;<br>
•	Приведение к размеру 32×32 пикселя.<br><br>

Объём: 300 изображений на каждую букву. Скрипт генерации написан на Python с использованием PIL и OpenCV. <br>

### 2.2 Расширение датасета<br>

В дальнейшем к датасету были добавлены скриншоты реальных изображений букв, размещённых в именованных папках. Применялись:<br>
•	Преобразования яркости, контрастности;<br>
•	Добавление шумов типа salt & pepper;<br>
•	Повороты на ±15°;<br>
•	Имитация освещения и движения.<br>
<br>
Суммарно датасет содержал примерно 25 000 изображений, по ~700 на каждую букву.<br>
<img width="1539" alt="Снимок экрана 2025-04-23 в 21 26 16" src="https://github.com/user-attachments/assets/8946d56e-d5c9-446c-9284-171b310c9cfd" />


## 3. Архитектура модели<br>

Была реализована и обучена сверточная нейронная сеть на PyTorch.<br>
Конфигурация:<br>

Слой	Назначение	Параметры <br>

Conv2D -    ReLU -    MaxPool  	Извлечение признаков	64 фильтра, ядро 3×3<br>
Conv2D -   ReLU -   MaxPool	  Углубление признаков	128 фильтров<br>
FC (128 -   64 -   36)  	Классификация	Полносвязные слои<br>
Параметры обучения:<br>

•	Оптимизатор: Adam, lr = 0.001, weight_decay = 1e-5<br>
•	Функция потерь: CrossEntropyLoss<br>
•	Аугментации: вращение, affine, jitter<br>
•	Эпох: 30<br>
•	Scheduler: StepLR (step=10, gamma=0.1)<br>

Общая точность на валидации: 97–100% для большинства классов.<br>
Минимальная точность: ~90% — у взаимозаменяемых букв (например, буквы, инверсные при повороте на 180°).<br>

<img width="1539" alt="Снимок экрана 2025-04-23 в 21 26 16" src="https://github.com/user-attachments/assets/fa9c03f7-e50b-4491-af43-cbc2bb1ced2e" />

<img width="1539" alt="Снимок экрана 2025-04-23 в 21 25 58" src="https://github.com/user-attachments/assets/804d207a-f0a7-48f5-892c-17a967563f78" />



## 4. Встраивание модели на Raspberry Pi<br>

Модель была экспортирована в .pth и интегрирована в управляющий скрипт, работающий на Raspberry Pi 4.<br>

Характеристики:<br>

•	Загружается в режиме eval, используется softmax для оценки уверенности;<br>
•	Обрабатываются изображения с камеры (PiCamera2 / USB);<br>
•	Используется трансформация: grayscale → resize → normalize.<br>


## 5. Управляющий скрипт<br>

### 5.1. Инициализация<br>

Скрипт реализует класс DroneController, инициализирующий:<br>

•	Подключение к Pixhawk по MAVLink (через UART /dev/ttyAMA0);<br>
![IMAGE 2025-04-24 12:53:40](https://github.com/user-attachments/assets/ad2e4ce8-bcff-4662-97cc-925a3d8df2c5)

•	Камеру (по конфигурации USE_PICAM);<br>
•	Модель и список меток классов;<br>
•	Систему логирования и журналирования.<br>



### 5.2. Режимы работы <br>

Режимы выбираются по значению RC-канала (по умолчанию — канал 6):<br>

Значение RC	Режим	Назначение<br>
<img width="1680" alt="Снимок экрана 2025-04-21 в 16 04 15" src="https://github.com/user-attachments/assets/0071a6a2-5460-4af3-a953-818bbcb9a00d" />


  < 1100	OFF	Телеметрия, мониторинг<br>
  1300–1700	LETTERS	Распознавание букв<br>
  1900	ARUCO	Поиск маркера и посадка<br>
  
Значения отфильтровываются: требуется минимум 3 совпадения подряд для подтверждения переключения режима (анти-дребезг).<br>



## 6. Распознавание и действия<br>

### 6.1. Распознавание букв<br>

•	Изображения сегментируются по контуру (на основе порога);<br>
•	Проверяются по размеру и форме (контуры квадратные, достаточно большие);<br>
•	На каждый фрагмент применяется модель;<br>
•	Если уверенность превышает CONFIDENCE_THRESHOLD, результат логируется.<br>


### 6.2. Обнаружение ArUco

•	Используется словарь DICT_6X6_250, маркер с ID=42;<br>
•	Проверка выполняется циклично с периодом 0.1 с;<br>
•	При обнаружении выполняется центрирование и команда на посадку. <br>
<img width="654" alt="Снимок экрана 2025-04-24 в 12 56 58" src="https://github.com/user-attachments/assets/0a004807-f831-438e-b568-c590b62cff15" />

•	Постепенное снижение тяги. Уменьшение значения PWM вплоть до значения 1000 с шагом 25 <br>
•	disarm<br>




## 7. Телеметрия<br>

В режиме OFF каждые 10 секунд контроллер отправляет в консоль по SSH:<br>

•	Высоту, скорость;<br>
•	GPS-координаты;<br>
•	Углы ориентации;<br>
•	Состояние батареи.<br>
Также осуществляется логирование в файл sla_controller.log.<br>


## 8. Безопасность и стабильность<br>

•	При потере связи MAVLink выполняется автоматическое переподключение;<br>
•	При переключении режима камера корректно освобождается и перезапускается;<br>
•	Все скрипты логируют свою активность;<br>
•	Реализован механизм ручного прерывания KeyboardInterrupt и остановки;<br>



## 9. Результаты тестирования<br>

•	Система корректно переключает режимы и выполняет посадку;<br>
•	Уверенность модели распознавания — свыше 90%;<br>
•	Удалённое управление через RC-пульт стабильно работает;<br>
•	ArUco-маркеры распознаются с расстояния до 15 м (при хорошей освещённости).<br>


<img width="1680" alt="Снимок экрана 2025-03-23 в 22 36 28" src="https://github.com/user-attachments/assets/47cd5fd4-cef8-4542-8d94-14efaca19d69" />

<img width="1680" alt="Снимок экрана 2025-04-23 в 21 23 00" src="https://github.com/user-attachments/assets/ccf4d566-7617-43f1-9e08-9b8da753b2dd" />

## 10. Заключение и планы по развитию<br>
Система продемонстрировала высокую точность и стабильность.<br>
Удалось реализовать:<br>

•	Полноценное управление режимами полёта;<br>
•	Распознавание букв армянского алфавита;<br>
•	Интеграцию с телеметрией и возможностью посадки;<br>
•	Логирование и автономную работу.<br>

Возможные улучшения:<br>
•	Модификация второй камеры USB (одна — для ArUco, вторая — для букв);<br>
•	Расширение датасета за счёт реальных съёмок;<br>
•	Оптимизация обработки изображения под Raspberry Pi;<br>
•	Разработка интерфейса для ручного управления и отладки.<br>





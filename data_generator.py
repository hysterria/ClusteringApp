from faker import *
import pandas as pd
import random
import tkinter as tk
from tkinter import ttk
from datetime import datetime

faker = Faker('ru_RU') # чтобы генерировало русские имена

def generate_fio():
    return faker.name()

def generate_age():
    return random.randint(18, 80)

def generate_gender():
    return random.choice(['Мужской', 'Женский'])

def generate_email(fio):
    # Преобразуем ФИО в email-подобный формат
    name_parts = fio.lower().split()
    if len(name_parts) >= 2:
        return f"{name_parts[0]}_{name_parts[1]}{random.randint(10,99)}@example.com"
    return faker.email()

def generate_phone():
    return f"+7{random.randint(900, 999)}{random.randint(1000000, 9999999)}"

def genrate_passport():
    series = random.randint(1000,9999)
    number = random.randint(100000, 999999)
    return f'{series} {number}'

cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань', 'Нижний Новгород', 'Челябинск', 'Самара', 'Омск', 'Ростов-на-Дону','Ижевск','Сарапул','Пермь','Нефтекамск','Анапа','Орёл']
def generate_route():
    from_city = random.choice(cities)
    to_city = random.choice([city for city in cities if city != from_city])
    return from_city, to_city

def generate_dates():
    departure_date = faker.date_time_between(start_date = '-2y',end_date = 'now')
    arrival_date = departure_date + pd.DateOffset(hours = random.randint(1,12))
    return departure_date, arrival_date

def generate_train():
    train_types = {
        'скорые поезда':range(1,150),
        'сезонные поезда':range(151,298),
        'пассажирские':range(301,450),
        'сезонные пассажирские':range(451,598),
        'скоростные':range(701,750),
        'высокоскоростные':range(751,788),
    }
    category = random.choice(list(train_types.keys()))
    train_numer = random.choice(train_types[category])
    return train_numer

def generate_wagon_and_seat(train):
    wagons_and_seats = {
        'Сапсан': ['1Р', '1В', '1С', '2С', '2В', '2E'],
        'Стриж': ['1Е', '1Р', '2С'],
        'Сидячий': ['1С', '1Р', '1В', '2Р', '2Е'],
        'Плацкарт': ['3Э'],
        'Купе': ['2Э'],
        'Люкс': ['1Б', '1Л'],
        'Мягкий': ['1А', '1И']
    }
    if train in range(1,585):
        train_type = random.choice(['Сидячий','Плацкарт','Купе','Люкс','Мягкий'])
    elif train in range(701,788):
        train_type = 'Сидячий'
    else:
        train_type = random.choice(['Сапсан','Стриж'])
    wagon = random.choice(wagons_and_seats[train_type])
    seat = random.randint(1,50)
    return f'{wagon}-{seat}'

def generate_price(type):
    base_price = random.randint(1000, 5000)  # Базовая цена
    if type in ['1Р', '1В', '1С','1Е','1B','2Р','1Б','1Л','1А','1И']:
        multiplier = 2
    if type in ['2С', '2В','2С','1Р','2Э']:
        multiplier = 1.5
    else:
        multiplier = 0.5
    return round(base_price * multiplier, 2)

def generate_ticket_status(departure_date):
    now = datetime.now()
    if departure_date < now:
        return random.choices(['Использован', 'Возвращен'], weights=[90, 10])[0]
    else:
        return random.choices(['Активен', 'Возвращен'], weights=[95, 5])[0]

def generate_payment_card(banks_prob,systems_prob):
    banks = ['Сбербанк', 'ВТБ', 'Альфа-Банк', 'Тинькофф']
    payment_systems = ['Visa', 'MasterCard', 'МИР']

    bank = random.choices(banks, weights=banks_prob, k=1)[0]
    system = random.choices(payment_systems, weights=systems_prob, k=1)[0]
    part1 = ''
    if system == 'МИР':
        if bank == 'Сбербанк':
            part1 = '2202'
        elif bank == 'Тинькофф':
            part1 = '2200'
        elif bank == 'ВТБ':
            part1 = '2204'
        else:
            part1 = '2206'
    elif system == 'MasterCard':
        if bank == 'Сбербанк':
            part1 = '5469'
        elif bank == 'Тинькофф':
            part1 = '5489'
        elif bank == 'ВТБ':
            part1 = '5443'
        else:
            part1 = '5406'
    else:
        if bank == 'Сбербанк':
            part1= '4276'
        elif bank == 'Тинькофф':
            part1 = '4277'
        elif bank == 'ВТБ':
            part1 = '4272'
        else:
            part1 = '4279'
    card_number = f"{part1} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
    return card_number

# Функция для генерации датасета
def generate_dataset(banks_prob, payment_systems_prob, n_rows=5000):
    data = []
    fios = []
    passports = []
    cards = []
    for _ in range(n_rows):
        fio = generate_fio()
        while fio in fios:
            fio = generate_fio()
        fios.append(fio)

        age = generate_age()
        gender = generate_gender()
        email = generate_email(fio)
        phone = generate_phone()

        passport = genrate_passport()
        while passport in passports:
            passport = genrate_passport()
        passports.append(passport)

        from_city, to_city = generate_route()

        departure, arrival = generate_dates()

        train_number = generate_train()
        train = f"{train_number}{random.choice(['A', 'B', 'C'])}"

        wagon_seat = generate_wagon_and_seat(train_number)

        price = generate_price(wagon_seat[:2])

        ticket_status = generate_ticket_status(departure)

        # Генерация платёжной карты
        payment_card = generate_payment_card(banks_prob, payment_systems_prob)
        while cards.count(payment_card) == 5:
            payment_card = generate_payment_card(banks_prob, payment_systems_prob)
        cards.append(payment_card)

        row = [
            fio, age, gender, email, phone,
            passport, from_city, to_city,
            departure.strftime('%Y-%m-%dT%H:%M'),
            arrival.strftime('%Y-%m-%dT%H:%M'),
            train, wagon_seat, price, ticket_status, payment_card
        ]
        data.append(row)

    columns = [
        'ФИО', 'Возраст', 'Пол', 'Email', 'Телефон',
        'Паспортные данные', 'Откуда', 'Куда',
        'Дата отъезда', 'Дата приезда', 'Рейс',
        'Вагон и место', 'Стоимость', 'Статус билета', 'Карта оплаты'
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('original_dataset.csv', index=False)
    print("Датасет успешно сохранен в файл 'original_dataset.csv'")


# Интерфейс tkinter с выбором вероятностей для банков и платежных систем
def start_interface():
    # Создаем окно
    root = tk.Tk()
    root.title("Настройка вероятностей банков и платежных систем")
    root.geometry("400x700")  # Устанавливаем размер окна

    # Функция для обработки выбора
    def on_submit():
        try:
            # Получение вероятностей банков
            prob_sberbank = int(sberbank_scale.get())
            prob_vtb = int(vtb_scale.get())
            prob_alfa = int(alfa_scale.get())
            prob_tinkoff = int(tinkoff_scale.get())
            total_banks = prob_sberbank + prob_vtb + prob_alfa + prob_tinkoff

            # Получение вероятностей платежных систем
            prob_visa = int(visa_scale.get())
            prob_mastercard = int(mastercard_scale.get())
            prob_mir = int(mir_scale.get())
            total_payment = prob_visa + prob_mastercard + prob_mir

            # Проверка на корректность суммы вероятностей
            errors = []
            if total_banks != 100:
                errors.append("Сумма вероятностей банков должна быть 100!")
            if total_payment != 100:
                errors.append("Сумма вероятностей платёжных систем должна быть 100!")

            if errors:
                error_label.config(text="\n".join(errors), fg="red")
            else:
                banks_prob = [prob_sberbank, prob_vtb, prob_alfa, prob_tinkoff]
                payment_systems_prob = [prob_visa, prob_mastercard, prob_mir]
                generate_dataset(banks_prob, payment_systems_prob)
                error_label.config(text="Датасет успешно создан!", fg="green")
        except ValueError:
            error_label.config(text="Ошибка: Введите корректные значения!", fg="red")

    # Метки и слайдеры для банков
    bank_frame = ttk.LabelFrame(root, text="Вероятности банков (%)")
    bank_frame.pack(padx=10, pady=10, fill="x")

    # Сбербанк
    ttk.Label(bank_frame, text="Сбербанк").pack(anchor='w', padx=10, pady=5)
    sberbank_scale = ttk.Scale(bank_frame, from_=0, to=100, orient="horizontal")
    sberbank_scale.set(40)  # Значение по умолчанию
    sberbank_scale.pack(fill="x", padx=10)
    sberbank_value = tk.IntVar(value=40)
    sberbank_scale.config(command=lambda val: sberbank_value.set(int(float(val))))
    ttk.Label(bank_frame, textvariable=sberbank_value).pack(anchor='e', padx=10)

    # ВТБ
    ttk.Label(bank_frame, text="ВТБ").pack(anchor='w', padx=10, pady=5)
    vtb_scale = ttk.Scale(bank_frame, from_=0, to=100, orient="horizontal")
    vtb_scale.set(30)
    vtb_scale.pack(fill="x", padx=10)
    vtb_value = tk.IntVar(value=30)
    vtb_scale.config(command=lambda val: vtb_value.set(int(float(val))))
    ttk.Label(bank_frame, textvariable=vtb_value).pack(anchor='e', padx=10)

    # Альфа-Банк
    ttk.Label(bank_frame, text="Альфа-Банк").pack(anchor='w', padx=10, pady=5)
    alfa_scale = ttk.Scale(bank_frame, from_=0, to=100, orient="horizontal")
    alfa_scale.set(20)
    alfa_scale.pack(fill="x", padx=10)
    alfa_value = tk.IntVar(value=20)
    alfa_scale.config(command=lambda val: alfa_value.set(int(float(val))))
    ttk.Label(bank_frame, textvariable=alfa_value).pack(anchor='e', padx=10)

    # Тинькофф
    ttk.Label(bank_frame, text="Тинькофф").pack(anchor='w', padx=10, pady=5)
    tinkoff_scale = ttk.Scale(bank_frame, from_=0, to=100, orient="horizontal")
    tinkoff_scale.set(10)
    tinkoff_scale.pack(fill="x", padx=10)
    tinkoff_value = tk.IntVar(value=10)
    tinkoff_scale.config(command=lambda val: tinkoff_value.set(int(float(val))))
    ttk.Label(bank_frame, textvariable=tinkoff_value).pack(anchor='e', padx=10)

    # Метки и слайдеры для платежных систем
    payment_frame = ttk.LabelFrame(root, text="Вероятности платёжных систем (%)")
    payment_frame.pack(padx=10, pady=10, fill="x")

    # Visa
    ttk.Label(payment_frame, text="Visa").pack(anchor='w', padx=10, pady=5)
    visa_scale = ttk.Scale(payment_frame, from_=0, to=100, orient="horizontal")
    visa_scale.set(50)
    visa_scale.pack(fill="x", padx=10)
    visa_value = tk.IntVar(value=50)
    visa_scale.config(command=lambda val: visa_value.set(int(float(val))))
    ttk.Label(payment_frame, textvariable=visa_value).pack(anchor='e', padx=10)

    # MasterCard
    ttk.Label(payment_frame, text="MasterCard").pack(anchor='w', padx=10, pady=5)
    mastercard_scale = ttk.Scale(payment_frame, from_=0, to=100, orient="horizontal")
    mastercard_scale.set(30)
    mastercard_scale.pack(fill="x", padx=10)
    mastercard_value = tk.IntVar(value=30)
    mastercard_scale.config(command=lambda val: mastercard_value.set(int(float(val))))
    ttk.Label(payment_frame, textvariable=mastercard_value).pack(anchor='e', padx=10)

    # MIR
    ttk.Label(payment_frame, text="МИР").pack(anchor='w', padx=10, pady=5)
    mir_scale = ttk.Scale(payment_frame, from_=0, to=100, orient="horizontal")
    mir_scale.set(20)
    mir_scale.pack(fill="x", padx=10)
    mir_value = tk.IntVar(value=20)
    mir_scale.config(command=lambda val: mir_value.set(int(float(val))))
    ttk.Label(payment_frame, textvariable=mir_value).pack(anchor='e', padx=10)

    # Кнопка для подтверждения
    submit_button = tk.Button(root, text="Создать датасет", command=on_submit)
    submit_button.pack(pady=20)

    # Поле для вывода ошибок и сообщений
    error_label = tk.Label(root, text="", fg="red", wraplength=380, justify="left")
    error_label.pack(pady=10)

    # Запуск интерфейса
    root.mainloop()


# Запуск интерфейса
if __name__ == "__main__":
    start_interface()

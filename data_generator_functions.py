from faker import *
import pandas as pd
import random
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

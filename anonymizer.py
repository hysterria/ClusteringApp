import csv
import io
import pandas as pd


def de_name(dict):
    # Заменяем ФИО на пол (уже есть отдельное поле Пол, но оставим для совместимости)
    dict['ФИО'] = dict['Пол'][0]  # 'М' или 'Ж'


def de_passport(dict):
    dict['Паспортные данные'] = '**** ******'


def de_marsh(dict):
    europe_cities = ['Москва', 'Санкт-Петербург', 'Казань', 'Нижний Новгород',
                     'Самара', 'Ростов-на-Дону', 'Ижевск', 'Сарапул', 'Пермь',
                     'Нефтекамск', 'Анапа', 'Орёл']

    dict['Откуда'] = 'Европа' if dict['Откуда'] in europe_cities else 'Азия'
    dict['Куда'] = 'Европа' if dict['Куда'] in europe_cities else 'Азия'


def de_train_type(dict):
    train_types = {
        'скорые поезда': range(1, 150),
        'сезонные поезда': range(151, 298),
        'пассажирские': range(301, 450),
        'сезонные пассажирские': range(451, 598),
        'скоростные': range(701, 750),
        'высокоскоростные': range(751, 788),
    }

    # Извлекаем номер поезда (удаляем последнюю букву)
    train_num = int(dict['Рейс'][:-1]) if dict['Рейс'][-1].isalpha() else int(dict['Рейс'])

    for train_type, num_range in train_types.items():
        if train_num in num_range:
            dict['Рейс'] = train_type
            break


def de_wagon(dict):
    dict['Вагон и место'] = '*****'


def de_price(dict):
    try:
        price = float(dict['Стоимость'])
    except:
        price = 0

    if price < 1000:
        dict['Стоимость'] = '< 1000'
    elif 1000 <= price < 2000:
        dict['Стоимость'] = '1000 - 2000'
    elif 2000 <= price < 4000:
        dict['Стоимость'] = '2000 - 4000'
    elif 4000 <= price < 5000:
        dict['Стоимость'] = '4000 - 5000'
    else:
        dict['Стоимость'] = '>=5000'


def de_card(dict):
    dict['Карта оплаты'] = '**** **** **** ****'


def de_date(dict):
    for date_field in ('Дата отъезда', 'Дата приезда'):
        if not dict[date_field] or pd.isna(dict[date_field]):
            dict[date_field] = 'Неизвестно'
            continue

        month = dict[date_field][5:7] if len(dict[date_field]) >= 7 else '00'

        if month in ('01', '02', '12'):
            dict[date_field] = 'Зима'
        elif month in ('03', '04', '05'):
            dict[date_field] = 'Весна'
        elif month in ('06', '07', '08'):
            dict[date_field] = 'Лето'
        elif month in ('09', '10', '11'):
            dict[date_field] = 'Осень'
        else:
            dict[date_field] = 'Неизвестно'


def de_age(dict):
    age = int(dict['Возраст']) if dict['Возраст'] else 0
    if age < 18:
        dict['Возраст'] = '<18'
    elif 18 <= age < 25:
        dict['Возраст'] = '18-24'
    elif 25 <= age < 35:
        dict['Возраст'] = '25-34'
    elif 35 <= age < 50:
        dict['Возраст'] = '35-49'
    elif 50 <= age < 65:
        dict['Возраст'] = '50-64'
    else:
        dict['Возраст'] = '65+'


def de_email(dict):
    dict['Email'] = '*****@*****.***'


def de_phone(dict):
    dict['Телефон'] = '+7**********'


def de_ticket_status(dict):
    # Оставляем как есть, так как это уже категориальное значение
    pass


# Основной процесс обезличивания
with io.open('train_tickets_dataset.csv', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    with io.open('new_train_tickets_dataset.csv', mode='w', encoding='utf-8', newline='') as newfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(newfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            de_name(row)
            de_age(row)
            # Поле 'Пол' уже содержит обобщенные данные, можно оставить
            de_email(row)
            de_phone(row)
            de_passport(row)
            de_marsh(row)
            de_train_type(row)
            de_wagon(row)
            de_price(row)
            de_card(row)
            de_date(row)
            # 'Статус билета' не обезличиваем, так как это уже категория

            writer.writerow(row)


# Функции для оценки k-анонимности остаются без изменений
def evaluate_data_utility(original_df: pd.DataFrame, anonymized_df: pd.DataFrame, quasi_identifiers: list):
    results = []

    for col in quasi_identifiers:
        original_unique = original_df[col].nunique()
        anonymized_unique = anonymized_df[col].nunique()
        utility_loss = (original_unique - anonymized_unique) / original_unique * 100
        results.append({
            'Квази-идентификатор': col,
            'Уникальные значения (исходные данные)': original_unique,
            'Уникальные значения (обезличенные данные)': anonymized_unique,
            'Потеря полезности (%)': round(utility_loss, 2)
        })

    return pd.DataFrame(results)


def calculate_k_anonymity_from_dict(data_list, quasi_identifiers):
    data = pd.DataFrame(data_list)

    grouped = data.groupby(quasi_identifiers).size().reset_index(name='count')

    k_anonymity = grouped['count'].min()

    worst_groups = grouped.sort_values(by='count', ascending=True).head(5)

    total_records = len(data)
    worst_groups['percentage'] = (worst_groups['count'] / total_records) * 100

    return k_anonymity, worst_groups


def remove_low_frequency_rows(data_list, quasi_identifiers, threshold):
    data = pd.DataFrame(data_list)

    grouped = data.groupby(quasi_identifiers).size().reset_index(name='count')

    frequent_groups = grouped[grouped['count'] >= threshold]

    filtered_data = data.merge(frequent_groups[quasi_identifiers], on=quasi_identifiers, how='inner')

    return filtered_data.to_dict(orient='records')


# Оценка k-анонимности
with io.open('new_train_tickets_dataset.csv', encoding='utf-8') as csvfile:
    new_reader = list(csv.DictReader(csvfile))

    print("Выберите квази-идентификаторы для оценки k-анонимности:")
    name = input('ФИО (Y/n): ')
    age = input('Возраст (Y/n): ')
    gender = input('Пол (Y/n): ')
    from_to = input('Маршрут (Y/n): ')
    dates = input('Даты (Y/n): ')
    rout = input('Рейс (Y/n): ')
    wagon = input('Вагон (Y/n): ')
    price = input('Стоимость (Y/n): ')
    card = input('Карта (Y/n): ')

    k_identifiers = []
    if name == 'Y':
        k_identifiers.append('ФИО')
    if age == 'Y':
        k_identifiers.append('Возраст')
    if gender == 'Y':
        k_identifiers.append('Пол')
    if from_to == 'Y':
        k_identifiers.append('Откуда')
        k_identifiers.append('Куда')
    if dates == 'Y':
        k_identifiers.append('Дата отъезда')
        k_identifiers.append('Дата приезда')
    if rout == 'Y':
        k_identifiers.append('Рейс')
    if wagon == 'Y':
        k_identifiers.append('Вагон и место')
    if price == 'Y':
        k_identifiers.append('Стоимость')
    if card == 'Y':
        k_identifiers.append('Карта оплаты')

    if k_identifiers:
        filtered_data = remove_low_frequency_rows(new_reader, k_identifiers, 10)
        with io.open('new_new_train_tickets_dataset.csv', mode='w', encoding='utf-8', newline='') as newfile:
            fieldnames = new_reader[0].keys() if new_reader else []
            writer = csv.DictWriter(newfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in filtered_data:
                writer.writerow(row)

        k_anonymity, worst_groups = calculate_k_anonymity_from_dict(filtered_data, k_identifiers)
        original_data = pd.read_csv("train_tickets_dataset.csv")
        anonymized_data = pd.read_csv("new_new_train_tickets_dataset.csv")
        utility_report = evaluate_data_utility(original_data, anonymized_data, k_identifiers)

        print("\nРезультаты оценки:")
        print("k-анонимность:", k_anonymity)
        print("\nТоп 5 самых маленьких групп:")
        print(worst_groups.to_string(index=False))
        print("\nОтчет о потере полезности данных:")
        print(utility_report.to_string(index=False))
    else:
        print("Не выбрано ни одного квази-идентификатора. Оценка k-анонимности не выполнена.")
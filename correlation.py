import pandas as pd
import matplotlib.pyplot as plt

movies = pd.read_csv(f"DisneyMoviesDataset.csv")
cpi = pd.read_csv(f"US CPI.csv")


def remove_empty(data, column_lst):
    """
    removes rows of given columns with missing data
    :param data: dataset
    :param column_lst: list of column names to remove rows from
    :return: dataset with columns that have all the data
    """
    # print(len(data))
    data = data.dropna(subset=column_lst)
    # print(data.isnull().mean())
    # print(len(data))
    return data


# print(remove_empty(movies, ['Budget (float)', 'Box office (float)']))

def extract_yearly_cpi(data):
    """
    calculates yearly cpi according to the average cpi of all days in each year
    :param data: cpi dataset per day
    :return: dict where key is year and value is average cpi
    """
    # converting to datetime
    data['Yearmon'] = pd.to_datetime(data['Yearmon'], format='%m-%d-%Y')

    data['year'] = data['Yearmon'].dt.year
    avg_yearly_cpi = data.groupby('year')['CPI'].mean()
    return avg_yearly_cpi.to_dict()


# print(extract_yearly_cpi(cpi))

def fix_dates(data):
    """
    fixes datetime object in the 1900s that were mistaken for 2000s
    :param data: dataset
    :return: dataset with correct datetime objects
    """
    data['Release date (datetime)'] = pd.to_datetime(data['Release date (datetime)'], format='%d-%m-%y')

    corrected_dates = []
    for index, row in data.iterrows():
        year = row['Release date (datetime)'].year
        if year > 2022:
            corrected_year = year - 2000 + 1900
            corrected_date = row['Release date (datetime)'].replace(year=corrected_year)
            corrected_dates.append(corrected_date)
        else:
            corrected_dates.append(row['Release date (datetime)'])

    data['Release date (datetime)'] = corrected_dates
    return data


# print(fix_dates(movies))

def normalise(cpi, field_name, data):
    """
    normalises monetary fields in dataset and adds them as a column to data
    :param cpi: cpi dataset
    :param field_name: field to normalise
    :param data: movie dataset
    :return: movie dataset with added columns of normalised values from field_lst
    """
    # converting to datetime and fixing dates, extracting yearly cpi
    data = fix_dates(data)
    data = remove_empty(data, ['Release date (datetime)', 'Budget (float)'])
    yearly_cpi = extract_yearly_cpi(cpi)

    # inflation rate of year i = (cpi 2021) / (cpi year i)
    # calculating inflation rate for each row
    inflation_rates = []
    for index, row in data.iterrows():
        year = row['Release date (datetime)'].year
        # print(year)
        cpi_year = yearly_cpi[year]
        inflation_rate = (yearly_cpi[2021] / cpi_year)
        inflation_rates.append(inflation_rate)

    data.loc[:, field_name + ' normalised'] = data[field_name] * inflation_rates
    return data


# print(normalise(cpi, 'Budget (float)', movies))
# print(normalise(cpi, 'Box office (float)', movies))

def budget_box_office(data, cpi):
    """
    maps out the budget vs box office, normalised to 2021
    :param data: movie dataset
    :param cpi: cpi dataset
    :return: scatterplot of budget and box office for each movie
    """
    data_c = data.copy()
    data_c = normalise(cpi, 'Budget (float)', data_c)
    data_c = normalise(cpi, 'Box office (float)', data_c)

    plt.scatter(data_c['Budget (float) normalised'], data_c['Box office (float) normalised'], marker='o', color='blue',
                alpha=0.5)

    plt.title("Scatter Plot of Budget vs Box Office".format('Budget (float)', 'Box office (float)'))
    plt.xlabel('Budget')
    plt.ylabel('Box Office')

    plt.grid(True)  # Add grid lines
    plt.show()

print(budget_box_office(movies, cpi))

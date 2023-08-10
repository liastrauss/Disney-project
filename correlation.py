import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

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


def budget_box_office(data, cpi, profit_line):
    """
    maps out the budget vs box office, normalised to 2021. in addition there's an option to show
    a line where budget=box office
    :param data: movie dataset
    :param cpi: cpi dataset
    :param k: number of clusters
    :param profit_line: if true, show line where budget = box office
    :return: dataset
    """
    data_c = data.copy()
    data_c = normalise(cpi, 'Budget (float)', data_c)
    data_c = normalise(cpi, 'Box office (float)', data_c)
    data_c = remove_empty(data_c, ['Budget (float)', 'Box office (float)'])

    # so we can see budget and box office in million dollars
    norm_budget = data_c['Budget (float) normalised'] / 1000000
    norm_box_office = data_c['Box office (float) normalised'] / 1000000

    plt.scatter(norm_budget, norm_box_office, marker='o', color='blue', alpha=0.5)

    # Perform k-means clustering
    X = np.column_stack((norm_budget, norm_box_office))
    kmeans = KMeans(n_clusters=3, random_state=0)  # You can adjust other parameters as needed
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    cluster_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    # Plot each data point with its assigned cluster color
    for i in range(3):
        plt.scatter(norm_budget[cluster_labels == i], norm_box_office[cluster_labels == i],
                    marker='o', color=cluster_colors[i], alpha=0.5, label=f'Cluster {i + 1}')

    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', label='Centroids')

    plt.title("Budget and Box Office, in million dollars".format('Budget (float)', 'Box office (float)'))
    plt.xlabel('Budget')
    plt.ylabel('Box Office')

    if profit_line:
        plt.plot([0, 500], [0, 500], linestyle='--', color='gray')

    plt.grid(True)  # Add grid lines
    plt.show()
    return data_c


# print(budget_box_office(movies, cpi, True))

def find_profit_margin(data, cpi, column_list, profitable):
    """
    finds the top 10 movies where the profit margin is either very high or very low
    :param data: movie dataset
    :param cpi: cpi dataset
    :param column_list: list of column names from dataset to find profit margin for.
    column_list[0] is coord x, column_list[1] is coord y
    :param profitable: true if i want to find movies that are max{box office-budget},
    false if i want to find movies that are max{budget-box office}
    :return: dictionary where key is the name of the movie, and value is the coordinates of
    fields in column_list.
    """
    data = budget_box_office(data, cpi, False)

    # Calculate the profit (box office - budget)
    data['profit'] = data[column_list[1]] - data[column_list[0]]

    # Sort the data by profit in descending order if looking for most profitable, otherwise in ascending order
    data_sorted = data.sort_values(by='profit', ascending=not profitable)

    # Get the top 10 movies based on profitability
    top_movies = data_sorted.head(10)

    # Create a dictionary to store the results
    result_dict = {}
    for index, row in top_movies.iterrows():
        movie_name = row['title']
        coordinates = [row[column_list[0]], row[column_list[1]]]
        result_dict[movie_name] = coordinates

    return result_dict

print(find_profit_margin(movies, cpi, ['Budget (float) normalised', 'Box office (float) normalised'], True))
print(find_profit_margin(movies, cpi, ['Budget (float) normalised', 'Box office (float) normalised'], False))



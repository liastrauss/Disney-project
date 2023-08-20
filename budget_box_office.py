import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

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
    # print(data.isnull().mean())
    data = data.dropna(subset=column_lst)
    # print(len(data))
    return data


# print(remove_empty(movies, ['imdb', 'rotten_tomatoes', 'metascore']))

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

def update_remakes(data):
    """
    finds all movies that are remakes and adds the year to their title for easier distinguishing
    :param data: movie dataset
    :return: dataset with remakes
    """
    # cleaning
    data = fix_dates(data)
    data = remove_empty(data, ['title', 'Release date (datetime)'])

    title_years = {}

    for index, row in data.iterrows():
        title = row['title']
        year = row['Release date (datetime)'].year

        if title in title_years:
            if year > title_years[title]:
                title_years[title] = year
        else:
            title_years[title] = year

    # Iterate through the dictionary and update movie titles
    for title, year in title_years.items():
        same_titles = data[data['title'] == title]
        if len(same_titles) > 1:
            for index, row in same_titles.iterrows():
                updated_title = title
                if row['Release date (datetime)'].year == year:
                    updated_title += f" ({year})"
                data.at[index, 'title'] = updated_title

    return data


# print(update_remakes(movies))


def normalise(cpi, field_name, data):
    """
    normalises monetary fields to 2021 in dataset and adds them as a column to data
    :param cpi: cpi dataset
    :param field_name: field to normalise
    :param data: movie dataset
    :return: movie dataset with added columns of normalised values from field_lst
    """
    # converting to datetime and fixing dates, extracting yearly cpi
    data = fix_dates(data)
    data = remove_empty(data, ['Release date (datetime)', field_name])
    yearly_cpi = extract_yearly_cpi(cpi)

    # inflation rate of year i = (cpi 2021/cpi year i)**(1/(2021-i)
    # calculating inflation rate for each row
    inflation_rates = []
    for index, row in data.iterrows():
        year = row['Release date (datetime)'].year
        # print(year)
        cpi_year = yearly_cpi[year]
        inflation_rate = (yearly_cpi[2021] / cpi_year) ** (1 / (2021 - year))
        inflation_rates.append(inflation_rate)

    data.loc[:, field_name + ' normalised'] = data[field_name] * inflation_rates
    return data


# print(normalise(cpi, 'Budget (float)', movies))
# print(normalise(cpi, 'Box office (float)', movies))


def budget_box_office(data, cpi, show, profit_line, regression):
    """
    maps out the budget vs box office, normalised to 2021. in addition there's an option to show
    a line where budget=box office and regression line
    :param data: movie dataset
    :param cpi: cpi dataset
    :param show: if true, show plot
    :param profit_line: if true, show line where budget = box office
    :param regression: if true, show regression line
    :return: shows plot
    """
    data_c = data.copy()
    data_c = normalise(cpi, 'Budget (float)', data_c)
    data_c = normalise(cpi, 'Box office (float)', data_c)
    data_c = remove_empty(data_c, ['Budget (float)', 'Box office (float)'])

    # so we can see budget and box office in million dollars
    norm_budget = data_c['Budget (float) normalised'] / 1000000
    norm_box_office = data_c['Box office (float) normalised'] / 1000000

    plt.scatter(norm_budget, norm_box_office, marker='o', color='blue', alpha=0.5)

    X = np.column_stack((norm_budget, norm_box_office))
    kmeans = KMeans(n_clusters=1, random_state=0)  # You can adjust other parameters as needed
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    cluster_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    for i in range(1):
        plt.scatter(norm_budget[cluster_labels == i], norm_box_office[cluster_labels == i],
                    marker='o', color=cluster_colors[i], alpha=0.5)

    # Plot the regression line for the whole plot
    if regression:
        coeffs = np.polyfit(norm_budget, norm_box_office, 1)
        reg_line = np.polyval(coeffs, norm_budget)
        plt.plot(norm_budget, reg_line, linestyle='-', color='red', alpha=0.7, label='Regression Line')

        r_squared = np.corrcoef(norm_box_office, reg_line)[0, 1] ** 2
        rmse = np.sqrt(mean_squared_error(norm_box_office, reg_line))
        print(f"R-squared for Regression: {r_squared:.4f}")
        print(f"RMSE for Regression: {rmse:.2f} million dollars")

    # plotting centeroids
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')

    plt.title("Budget and Box Office, in million dollars")
    plt.xlabel('Budget')
    plt.ylabel('Box Office')

    if profit_line:
        plt.plot([0, 500], [0, 500], linestyle='--', color='gray', label='Profitability line')

    plt.grid(True)  # Add grid lines
    plt.legend()

    if show:
        plt.show()
    return data_c


# print(budget_box_office(movies, cpi, True, True, True))


def find_profit_margin(data, cpi, column_list):
    """
    Finds the top 10 movies with the highest profit margin and visualizes the results
    :param data: movie dataset
    :param cpi: cpi dataset
    :param column_list: list of column names from dataset to find profit margin for.
    column_list[0] is coord x, column_list[1] is coord y
    :return: Double bar chart depicting this
    """
    data = budget_box_office(data, cpi, False, False, False)
    data = update_remakes(data)

    # calculating the profit (box office - budget)
    data['profit'] = data[column_list[1]] - data[column_list[0]]

    # sorting by the absolute difference between budget and box office in descending order
    data_sorted = data.sort_values(by='profit', ascending=False)
    top_movies = data_sorted.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    movie_names = top_movies['title']
    budget_values = top_movies[column_list[0]] / 1000000
    box_office_values = top_movies[column_list[1]] / 1000000

    bar_width = 0.35
    index = range(len(movie_names))

    budget_bars = ax.bar(index, budget_values, bar_width, label='Budget')
    box_office_bars = ax.bar([i + bar_width for i in index], box_office_values, bar_width, label='Box Office')

    ax.set_xlabel('Movies')
    ax.set_ylabel('Values (in million dollars)')

    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(movie_names, rotation=45, ha='right')
    plt.xticks(wrap=True)
    ax.legend()

    plt.tight_layout()
    plt.show()


# print(find_profit_margin(movies, cpi, ['Budget (float) normalised', 'Box office (float) normalised']))


def ten_highest(data, cpi, field):
    """
    Finds the 10 highest values of a specified field in the dataset.

    :param data: movie dataset
    :param cpi: cpi dataset
    :param field: field to find 10 highest values for
    :return: Bar chart visualizing the top 10 movies with the highest field value
    """
    data_c = data.copy()
    data_c = normalise(cpi, field, data_c)
    data_c = update_remakes(data_c)

    top_10 = data_c.nlargest(10, field + ' normalised')

    plt.figure(figsize=(10, 6))
    plt.bar(top_10['title'], top_10[field + ' normalised'] / 1000000, color='blue')
    plt.xlabel('Movie')
    field = field.replace('(float)', '')
    plt.ylabel(field + ' (in million dollars)')
    plt.title(f'Top 10 Movies with Highest {field}')
    plt.xticks(rotation=45, ha='right', wrap=True)  # Wrap movie names
    plt.tight_layout()
    plt.show()

# print(ten_highest(movies, cpi, 'Budget (float)'))
# print(ten_highest(movies, cpi, 'Box office (float)'))

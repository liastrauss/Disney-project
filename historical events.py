import pandas as pd
import matplotlib.pyplot as plt
from budget_box_office import *

movies = pd.read_csv(f"DisneyMoviesDataset.csv")


def historical(data):
    """
    Cleaning the data and organizing it. Counting the number of films released each year
    :param data: movie dataset
    :return: A graph showing the amount of movies per year
    """
    data = remove_empty(data, ["Release date (datetime)"])
    data = fix_dates(data)

    # Extract year directly
    data['Release Year'] = data['Release date (datetime)'].dt.year
    year_counts = data['Release Year'].value_counts().sort_index()

    # Create a range of years from the data
    min_year = min(data['Release Year'])
    max_year = max(data['Release Year'])
    all_years = range(min_year, max_year + 1)

    # Calculate counts for each year, including zero counts
    year_counts = year_counts.reindex(all_years, fill_value=0)

    plt.figure(figsize=(10, 6))
    year_counts.plot(kind='bar', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies Released')
    plt.title('Distribution of Movies Released by Year')
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.tight_layout()
    plt.show()


historical(movies)

from budget_box_office import *

movies = pd.read_csv(f"DisneyMoviesDataset.csv")
cpi = pd.read_csv(f"US CPI.csv")


def monthly_success(data, cpi):
    """
    calculates relative success of movies according to month
    :param data: movies dataset
    :param cpi: cpi dataset
    :return: histogram that shows the average box office per month
    """
    # cleaning data
    data = normalise(cpi, "Box office (float)", data)
    data = remove_empty(data, ["Box office (float)", "Release date (datetime)"])
    data = fix_dates(data)

    data['Release Month'] = data['Release date (datetime)'].dt.strftime('%b')
    # print(data['Release Month'])

    monthly_avg_box_office = data.groupby('Release Month')['Box office (float) normalised'].mean() / 1000000
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg_box_office = monthly_avg_box_office.reindex(month_order)
    monthly_avg_box_office.plot(kind='bar', color='blue')
    plt.title('Movie Success according to Release Month')
    plt.xlabel('Month')
    plt.ylabel('Average Box Office, in million dollars')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# print(monthly_success(movies, cpi))

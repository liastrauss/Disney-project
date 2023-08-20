from budget_box_office import *

movies = pd.read_csv(f"DisneyMoviesDataset.csv")
cpi = pd.read_csv(f"US CPI.csv")


def avg_monthly_success(data, cpi):
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
    plt.title('Average Movie Success by Release Month')
    plt.xlabel('Month')
    plt.ylabel('Average Box Office (in million dollars)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

print(avg_monthly_success(movies, cpi))


def monthly_success(data, cpi):
    """
    Calculates relative success of movies according to month and visualizes it as a scatter plot.

    :param data: movies dataset
    :param cpi: cpi dataset
    :return: Scatter plot that shows the box office per movie for each month
    """
    # cleaning data
    data = normalise(cpi, "Box office (float)", data)
    data = remove_empty(data, ["Box office (float)", "Release date (datetime)"])
    data = fix_dates(data)

    data['Release Month'] = data['Release date (datetime)'].dt.strftime('%b')

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data['Release Month'] = pd.Categorical(data['Release Month'], categories=month_order, ordered=True)

    # Sort data by release month
    data_sorted = data.sort_values('Release Month')

    plt.figure(figsize=(10, 6))

    for month in data_sorted['Release Month'].unique():
        month_data = data_sorted[data_sorted['Release Month'] == month]
        plt.scatter([month] * len(month_data), month_data['Box office (float) normalised'] / 1000000,
                    alpha=0.7)

    # Plot the mean box office line for each month
    monthly_mean_box_office = data_sorted.groupby('Release Month')['Box office (float) normalised'].mean() / 1000000
    plt.plot(monthly_mean_box_office.index, monthly_mean_box_office, marker='o', color='red', linestyle='dashed',
             label='Mean Box Office')

    plt.title('Movie Box Office by Release Month')
    plt.xlabel('Release Month')
    plt.ylabel('Box Office, in million dollars')
    plt.xticks(rotation=45)  # Rotate x-axis tick labels for better visibility
    plt.legend()
    plt.tight_layout()
    plt.show()


print(monthly_success(movies, cpi))

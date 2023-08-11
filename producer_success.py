from monthly_success import *

movies = pd.read_csv(f"DisneyMoviesDataset.csv")


# def find_disney(data):
#     data = remove_empty(data, 'Produced by')
#     for prod in data['Produced by']:
#         if 'Disney' in prod:
#             print(prod)


# print(find_disney(movies))

def get_comb_rating(data):
    """
    calculates the combined rating of movie according to rotten tomatoes and imdb and adds as column
    :param data: movie dataset
    :return: dataset with new column of combined rating
    """
    data = remove_empty(data, ['imdb', 'rotten_tomatoes'])
    data['rotten_tomatoes'] = data['rotten_tomatoes'].apply(lambda x: float(x.strip('%')) / 10)
    data['imdb'] = data['imdb'].astype(float)
    # print(data['rotten_tomatoes'])
    # print(data['imdb'])

    data.loc[:, 'combined rating'] = (data['imdb'] + data['rotten_tomatoes']) / 2
    return data


# print(get_avg_rating(movies))

def producer_success(data):
    """
    calculates rating of movies according to whether they were produced by walt disney
    :param data: movie dataset
    :return: histogram showing walt disney's success as a producer vs. other producers
    """
    # cleaning data
    data = get_comb_rating(data)
    data = remove_empty(data, 'Produced by')

    disney_movies = data[data['Produced by'].str.contains('Disney', case=False, na=False)]
    non_disney_movies = data[~data['Produced by'].str.contains('Disney', case=False, na=False)]

    mean_disney_rating = disney_movies['combined rating'].mean()
    mean_non_disney_rating = non_disney_movies['combined rating'].mean()

    plt.bar(['Disney', 'Non-Disney'], [mean_disney_rating, mean_non_disney_rating], color=['blue', 'orange'])
    plt.title('Mean Combined Ratings Comparison')
    plt.xlabel('Producer')
    plt.ylabel('Mean Combined Rating')
    plt.ylim(0, 10)
    plt.show()

print(producer_success(movies))




def most_successful_producer(data):
    """
    maps out the success of each individual producer according to rating
    :param data:
    :return:
    """
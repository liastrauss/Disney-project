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


# print(producer_success(movies))

def clean_producers(data):
    """
    cleans producers column in dataset
    :param data: movie dataset
    :return: dataset with clean column
    """
    cleaned_producers = []
    for producer in data['Produced by']:
        if isinstance(producer, str):
            # producer full name
            cleaned_producers.append(producer.strip())
        elif isinstance(producer, list):
            # list of producers
            cleaned_producers.append(', '.join([p.strip() for p in producer]))
        elif isinstance(producer, tuple):
            # multiple producers as a comma-separated string
            cleaned_producers.append(', '.join([p.strip() for p in producer[0].split(',')]))
        else:
            cleaned_producers.append(None)

    data['Produced by (clean)'] = cleaned_producers
    # print(cleaned_producers)
    return data


# print(clean_producers(movies))


def detailed_producer_success(data):
    """
    calculates rating of movies according to whether they were produced by walt disney alone, by
    walt disney and others, or without walt disney
    :param data: movie dataset
    :return: histogram mapping it out
    """
    # cleaning
    data = get_comb_rating(data)
    data = remove_empty(data, ['Produced by', 'combined rating'])
    data = clean_producers(data)

    disney_alone = data[data['Produced by (clean)'].str.lower() == 'walt disney']
    disney_and_others = data[data['Produced by (clean)'].str.contains('walt disney', case=False) & ~(
            data['Produced by (clean)'].str.lower() == 'walt disney')]
    without_disney = data[~data['Produced by (clean)'].str.contains('walt disney', case=False)]

    # mean combined ratings for each group
    mean_disney_alone = disney_alone['combined rating'].mean()
    mean_disney_and_others = disney_and_others['combined rating'].mean()
    mean_without_disney = without_disney['combined rating'].mean()

    # Create a histogram
    # plt.figure(figsize=(10, 6))
    plt.bar(['Disney Alone', 'Disney and Others', 'Without Disney'],
            [mean_disney_alone, mean_disney_and_others, mean_without_disney],
            color=['blue', 'orange', 'green'])
    plt.title('Detailed Producer Success Comparison')
    plt.xlabel('Producer Group')
    plt.ylabel('Mean Combined Rating')
    plt.ylim(0, 10)
    plt.show()


# print(detailed_producer_success(movies))

def ten_producers(data, successful):
    """
    shows the top 10 most / least successful producers according to movie rating
    :param data: movie dataset
    :param successful: true for most successful, false for least successful
    :return: histogram of top 10 most / least successful producers according to rating
    """
    # cleaning
    data = get_comb_rating(data)
    data = clean_producers(data)
    data = remove_empty(data, ['combined rating', 'Produced by (clean)'])

    # creating a list of tuples with producer and rating
    producer_ratings = []
    for idx, row in data.iterrows():
        producers = row['Produced by (clean)'].split(', ')
        for producer in producers:
            cleaned_producer = producer.strip('[]')  # Remove square brackets
            producer_ratings.append((cleaned_producer, row['combined rating']))

    # creating a dataframe from the list of tuples
    producer_ratings_df = pd.DataFrame.from_records(producer_ratings, columns=['Producer', 'Rating'])

    # calculating the average rating for each producer
    avg_producer_ratings = producer_ratings_df.groupby('Producer')['Rating'].mean()

    # sorting the avg_producer_ratings based on success (ascending if successful=False)
    sorted_producer_ratings = avg_producer_ratings.sort_values(ascending=not successful)

    # getting top 10 producers
    top_producers = sorted_producer_ratings.head(10)

    plt.figure(figsize=(10, 6))
    top_producers.plot(kind='bar', color='blue')
    plt.ylim(0, 10)
    plt.title('Top 10 ' + ('Most' if successful else 'Least') + ' Successful Producers')
    plt.xlabel('Producer')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# print(ten_producers(movies, True))
# print(ten_producers(movies, False))
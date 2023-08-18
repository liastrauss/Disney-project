from monthly_success import *

movies = pd.read_csv(f"DisneyMoviesDataset.csv")


def make_rating_uniform(data):
    """
    Modifies ratings such that they are all float and between 1-10.
    :param data: Movie dataset
    :return: Dataset with uniform ratings
    """
    data['imdb'] = pd.to_numeric(data['imdb'], errors='coerce')
    data['metascore'] = pd.to_numeric(data['metascore'], errors='coerce') / 10
    data['rotten_tomatoes'] = pd.to_numeric(data['rotten_tomatoes'].str.replace('%', ''), errors='coerce') / 10

    return data


# print(make_rating_uniform(movies))

def clean_rating(data):
    """
    inserts the average rating of each movie for empty cells
    :param data: movie dataset
    :return: dataset with full rating information
    """
    # making uniform
    data = make_rating_uniform(data)

    data['average_rating'] = data[['imdb', 'metascore', 'rotten_tomatoes']].apply(
        lambda row: np.mean([value for value in row if pd.notnull(value)]),
        axis=1
    )

    # replacing empty cells with row average
    data['imdb'] = data.apply(
        lambda row: row['average_rating'] if not row.isnull().all() and pd.isnull(row['imdb']) else row['imdb'], axis=1)
    data['metascore'] = data.apply(
        lambda row: row['average_rating'] if not row.isnull().all() and pd.isnull(row['metascore']) else row[
            'metascore'], axis=1)
    data['rotten_tomatoes'] = data.apply(
        lambda row: row['average_rating'] if not row.isnull().all() and pd.isnull(row['rotten_tomatoes']) else row[
            'rotten_tomatoes'], axis=1)

    data.drop(columns=['average_rating'], inplace=True)

    # print(data['imdb'], data['metascore'], data['rotten_tomatoes'])
    return data


# print(clean_rating(movies))


def get_comb_rating(data):
    """
    calculates the combined rating of movie and adds as column
    :param data: movie dataset
    :return: dataset with new column of combined rating
    """
    # cleaning
    data = clean_rating(data)
    data = remove_empty(data, ['imdb', 'metascore', 'rotten_tomatoes'])

    data['combined rating'] = data[['imdb', 'metascore', 'rotten_tomatoes']].mean(axis=1)

    return data


# print(get_comb_rating(movies))


def producer_success(data):
    """
    calculates rating of movies according to whether they were produced by walt disney
    :param data: movie dataset
    :return: histogram showing walt disney's success as a producer vs. other producers for each rating
    """
    # cleaning data
    # data = get_comb_rating(data)
    # data = remove_empty(data, ['Produced by', 'combined rating'])
    data = clean_rating(data)
    data = remove_empty(data, ['Produced by', 'imdb', 'metascore', 'rotten_tomatoes'])

    # roy e. disney doesn't count as walt disney
    disney_movies = data[data['Produced by'].str.contains('|'.join(['Disney']), case=False, na=False)]
    roy_disney_movies = data[data['Produced by'].str.contains('Roy E. Disney', case=False, na=False)]
    disney_movies = disney_movies[~disney_movies['title'].isin(roy_disney_movies['title'])]

    # Calculate average ratings for Disney and other producers
    disney_avg_ratings = disney_movies[['imdb', 'metascore', 'rotten_tomatoes']].mean()
    other_movies = data[~data['title'].isin(disney_movies['title'])]
    other_avg_ratings = other_movies[['imdb', 'metascore', 'rotten_tomatoes']].mean()

    # Create a bar plot for average ratings
    rating_columns = ['IMDB', 'Metascore', 'Rotten Tomatoes']
    x = range(len(rating_columns))

    plt.bar(x, disney_avg_ratings, width=0.4, label='Disney')
    plt.bar([i + 0.4 for i in x], other_avg_ratings, width=0.4, label='Other Producers')

    plt.xlabel('Rating Categories')
    plt.ylabel('Average Rating')
    plt.title('Average Ratings for Disney vs. Other Producers')
    plt.xticks([i + 0.2 for i in x], rating_columns)
    plt.ylim(0, 10)
    plt.legend()
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

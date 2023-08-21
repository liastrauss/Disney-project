from monthly_success import *
import networkx as nx
import re

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
    :return: side-by-side bar plots showing walt disney's success as a producer vs. other producers for each rating
    """
    # cleaning data
    data = clean_rating(data)
    data = remove_empty(data, ['Produced by', 'imdb', 'metascore', 'rotten_tomatoes'])

    # seperating disney and non disney
    disney_movies = data[data['Produced by'].str.contains('|'.join(['Disney']), case=False, na=False)]
    roy_disney_movies = data[data['Produced by'].str.contains('Roy E. Disney', case=False, na=False)]
    disney_movies = disney_movies[~disney_movies['title'].isin(roy_disney_movies['title'])]
    non_disney_movies = data[~data['title'].isin(disney_movies['title'])]

    # Calculate average ratings for Disney movies
    disney_avg_ratings = {
        'imdb': disney_movies['imdb'].mean(),
        'metascore': disney_movies['metascore'].mean(),
        'rotten_tomatoes': disney_movies['rotten_tomatoes'].mean()
    }

    # Calculate average ratings for non-Disney movies
    non_disney_avg_ratings = {
        'imdb': non_disney_movies['imdb'].mean(),
        'metascore': non_disney_movies['metascore'].mean(),
        'rotten_tomatoes': non_disney_movies['rotten_tomatoes'].mean()
    }

    # Create a DataFrame for the average ratings
    avg_ratings_df = pd.DataFrame([disney_avg_ratings, non_disney_avg_ratings], index=['Disney', 'Non-Disney'])

    # Plotting
    avg_ratings_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Ratings for Disney vs Non-Disney Movies')
    plt.ylabel('Average Rating')
    plt.xlabel('Movie Type')
    plt.xticks(rotation=0)
    plt.legend(title='Rating Category')
    plt.ylim(0, 10)

    plt.show()


# print(producer_success(movies))


def clean_producers(data):
    """
    Cleans producers column in the dataset
    :param data: movie dataset
    :return: dataset with clean column
    """
    data = remove_empty(data, ['Produced by'])
    cleaned_producers = []

    # Regex pattern for characters to remove
    pattern = r'\([^)]*\)|\[|\]|\'|Associate Producer:'

    for producer in data['Produced by']:
        if isinstance(producer, str):
            # Apply regex pattern and strip
            producer = re.sub(pattern, '', producer).strip()
            # producer = re.sub(r'\b\w\. ', '', producer)
            cleaned_producers.append(producer)
        elif isinstance(producer, list):
            # List of producers
            cleaned_producers.append(', '.join([p.strip() for p in producer]))
        elif isinstance(producer, tuple):
            # Multiple producers as a comma-separated string
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
    # Cleaning data
    data = clean_rating(data)
    data = remove_empty(data, ['Produced by', 'imdb', 'metascore', 'rotten_tomatoes'])
    data = clean_producers(data)
    # data = filter_producers(data)

    disney_alone = data[data['Produced by (clean)'].str.lower() == 'walt disney']
    disney_and_others = data[data['Produced by (clean)'].str.contains('walt disney', case=False) & ~(
            data['Produced by (clean)'].str.lower() == 'walt disney')]
    without_disney = data[~data['Produced by (clean)'].str.contains('walt disney', case=False)]

    # Calculate average ratings for each producer category
    disney_alone_avg_ratings = disney_alone[['imdb', 'metascore', 'rotten_tomatoes']].mean()
    disney_and_others_avg_ratings = disney_and_others[['imdb', 'metascore', 'rotten_tomatoes']].mean()
    without_disney_avg_ratings = without_disney[['imdb', 'metascore', 'rotten_tomatoes']].mean()

    # Create bar plots for average ratings
    producer_categories = ['Disney Alone', 'Disney and Others', 'Without Disney']
    x = range(len(producer_categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rating_col in enumerate(['imdb', 'metascore', 'rotten_tomatoes']):
        ax.bar([pos + width * i for pos in x],
               [disney_alone_avg_ratings[rating_col], disney_and_others_avg_ratings[rating_col],
                without_disney_avg_ratings[rating_col]], width, label=rating_col)

    ax.set_xlabel('Producer Categories')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Ratings for Different Producer Categories')
    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(producer_categories)
    ax.set_ylim(0, 10)  # Set y-axis limit to 0-10
    ax.legend(loc='upper left')

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
    # data = filter_producers(data)

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
def prod_social_graph(data):
    data = clean_producers(data)
    data = remove_empty(data, ['Produced by (clean)'])

    G = nx.Graph()  # Create an empty graph

    # Add edges between producers who have worked together
    for producers_list in data['Produced by (clean)']:
        if producers_list is not None:
            producers = producers_list.split(', ')
            for i in range(len(producers)):
                producers[i] = producers[i].replace(',', '').strip()  # Corrected line
                for j in range(i + 1, len(producers)):
                    if producers[i] != producers[j]:  # Check for equality
                        # Check if the edge already exists before adding
                        if not G.has_edge(producers[i], producers[j]):
                            G.add_edge(producers[i], producers[j])

    # Draw the social graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Position nodes using a spring layout algorithm
    nx.draw(G, pos, with_labels=True, node_size=10, font_size=8, font_color='black')

    nx.draw_networkx_nodes(G, pos, nodelist=['Walt Disney'], node_color='red', node_size=20)

    plt.title("Producer Social Graph")
    plt.show()


# print(prod_social_graph(movies))

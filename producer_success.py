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


def clean_producers(data):
    """
    Cleans producers column in the dataset
    :param data: movie dataset
    :return: dataset with clean column
    """
    data = remove_empty(data, ['Produced by'])
    cleaned_producers = []

    # Regex pattern for characters to remove
    pattern = r'\([^)]*\)|\[|\]|\'|Associate Producer:|Executive:'

    for producer in data['Produced by']:
        if isinstance(producer, str):
            # Apply regex pattern and strip
            producer = re.sub(pattern, '', producer).strip()
            producer = re.sub(r'\b[A-Z]\.?\s', '', producer)
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
    # print(data['Produced by (clean)'].unique())
    return data


# print(clean_producers(movies))


def should_keep(producers, producers_to_remove):
    """
    Determines whether a movie should be kept based on its producers
    :param producers: List of producers for a movie
    :param producers_to_remove: List of producers to be removed
    :return: True if the movie should be kept, False otherwise
    """
    return any(producer not in producers_to_remove for producer in producers)


def filter_producers(data):
    """
    Filters movies based on individual producers who have produced less than 10 movies
    :param data: movie dataset
    :return: dataset with filtered movies, list of producers and how many movies they produced
    """
    # Cleaning and preprocessing
    data = clean_producers(data)
    data = remove_empty(data, ['Produced by (clean)'])

    # Remove any trailing commas in producers' names
    data['Produced by (clean)'] = data['Produced by (clean)'].str.replace(r',\s*$', '', regex=True)

    # Count the number of movies produced by each producer
    producer_counts = data['Produced by (clean)'].str.split(', ').explode().value_counts()

    # Identify producers with less than 6 movies
    producers_to_remove = set(producer_counts[producer_counts < 6].index)

    # Filter movies using the should_keep function
    data['Filtered Producers'] = data['Produced by (clean)'].str.split(', ').apply(
        lambda producers: should_keep(producers, producers_to_remove))
    data = data[data['Filtered Producers']]

    # Clean up the temporary columns
    data.drop(columns=['Filtered Producers'], inplace=True)

    return data, producer_counts


# print(filter_producers(movies))


def ten_producers(data):
    """
    Shows the top 10 most successful producers according to average combined rating
    :param data: movie dataset
    :return: shows bar graph of top producers, returns list of top producers
    """
    # Cleaning and filtering steps
    data = get_comb_rating(data)
    data = clean_producers(data)
    data, producer_counts = filter_producers(data)
    data = remove_empty(data, ['combined rating', 'Produced by (clean)'])

    successful_producers = producer_counts[producer_counts >= 7].index

    producer_avg_ratings = {}  # Dictionary to store producer's average combined rating

    # Calculate average combined rating for each successful producer
    for producer in successful_producers:
        producer_data = data[data['Produced by (clean)'].str.contains(producer)]
        avg_rating = producer_data['combined rating'].mean()
        producer_avg_ratings[producer] = avg_rating

    # Convert the dictionary to a DataFrame
    avg_rating_df = pd.DataFrame(list(producer_avg_ratings.items()), columns=['Producer', 'Average Rating'])

    # Add the movie count to the DataFrame
    avg_rating_df['Movie Count'] = avg_rating_df['Producer'].apply(lambda producer: producer_counts.get(producer, 0))

    # Sort the DataFrame by average rating in descending order and select the top 10 producers
    top_producers = avg_rating_df.sort_values(by='Average Rating', ascending=False).head(10)

    # Plot a histogram of top 10 producers and their average ratings
    plt.figure(figsize=(10, 6))
    plt.bar(top_producers['Producer'] + ' (' + top_producers['Movie Count'].astype(str) + ')',
            top_producers['Average Rating'])
    plt.xlabel('Producer (no. of movies produced)')
    plt.ylabel('Average Combined Rating')
    plt.title('Top 10 Most Successful Producers by Average Combined Rating')
    plt.ylim(1, 10)  # Set y-axis range between 1 and 10
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return top_producers['Producer'].tolist()


# print(ten_producers(movies))


def prod_collab_graph(data):
    """
    plots a collaboration graph of notable producers, emphasizing 10 most successful
    :param data: movie dataset
    :return: collaboration graph
    """
    data = clean_producers(data)
    data = remove_empty(data, ['Produced by (clean)'])
    data, producer_count = filter_producers(data)
    top_producers = ten_producers(data)

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes for producers with 6 or more movies
    for producer, count in producer_count.items():
        if count >= 6:
            G.add_node(producer)

    # Add edges between producers listed together in "Produced by (clean)"
    for row in data.iterrows():
        producers = row[1]['Produced by (clean)'].split(', ')
        for i in range(len(producers)):
            for j in range(i + 1, len(producers)):
                if G.has_node(producers[i]) and G.has_node(producers[j]):
                    G.add_edge(producers[i], producers[j])

    node_colours = ['red' if node in top_producers else 'blue' for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G, 1.1)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=10, node_color=node_colours)
    plt.title('Notable Producers Social Graph')
    plt.show()

# print(prod_collab_graph(movies))

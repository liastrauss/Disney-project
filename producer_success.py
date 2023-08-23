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


def prod_social_graph(data):
    """
    plots a social graph of notable producers, emphasizing 10 most successful
    :param data: movie dataset
    :return: social graph
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


print(prod_social_graph(movies))

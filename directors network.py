from monthly_success import *
movies = pd.read_csv(f"DisneyMoviesDataset.csv")
import networkx as nx

def clean_directors(data):
    """
    cleans directors column in dataset
    :param data: movie dataset
    :return: dataset with clean column
    """
    cleaned_directors = []
    for director in data['Directed by']:
        if isinstance(director, str):
            # director full name
            cleaned_directors.append(director.strip())
        elif isinstance(director, list):
            # list of directors
            cleaned_directors.append(', '.join([d.strip() for d in director]))
        elif isinstance(director, tuple):
            # multiple directors as a comma-separated string
            cleaned_directors.append(', '.join([d.strip() for d in director[0].split(',')]))
        else:
            cleaned_directors.append(None)

    data['Directed by (clean)'] = cleaned_directors
    return data


def directors_collab_graph(data):
    """
    plots a collaboration graph of the directors
    :param data: movie dataset
    :return: collaboration graph
    """
    data = clean_directors(data)
    data = remove_empty(data, ['Directed by (clean)'])

    # Create an undirected graph
    G = nx.Graph()

    # Add edges between directors listed together in "Directed by (clean)"
    for row in data.iterrows():
        directors = row[1]['Directed by (clean)'].split(', ')
        for i in range(len(directors)):
            for j in range(i + 1, len(directors)):
                if G.has_node(directors[i]) and G.has_node(directors[j]):
                    G.add_edge(directors[i], directors[j])

    # node_colours = ['red' if node in top_directors else 'blue' for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G, 1.1)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=10)
    plt.title('Directors Collaboration Graph') 
    plt.show()


print(directors_collab_graph(movies))
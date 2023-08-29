from monthly_success import *
movies = pd.read_csv(f"DisneyMoviesDataset.csv")

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

    data['Produced by (clean)'] = cleaned_directors
    print(cleaned_directors)
    return data

clean_directors(movies)
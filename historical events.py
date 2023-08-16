import pandas as pd
movies = pd.read_csv(f"DisneyMoviesDataset.csv")

def historical():
    print(movies)

historical()
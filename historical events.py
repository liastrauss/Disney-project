import pandas as pd
from budget_box_office import *

movies = pd.read_csv(f"DisneyMoviesDataset.csv")

def historical(data):
    data = remove_empty(data, ["Release date (datetime)"])
    data = fix_dates(data)

    data['Release Year'] = data['Release date (datetime)'].dt.strftime('%Y')
    # print(data['Release Year'])

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.hist(data['Release Year'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.title('Year Distribution')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate X-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout for better label visibility
    plt.show()



historical(movies)
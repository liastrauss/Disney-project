import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np


# Data cleaning
###############

reviews_data = pd.read_csv('DisneylandReviews.csv',
                           encoding="cp1252", na_values=['missing'])
# here we delete 538~ that containe "missing" string
reviews_data = reviews_data.dropna(how='any')


# word clouds
#############

# Combining all the reviews
reviews_text = ' '.join(
    str(review) for review in reviews_data['Review_Text'] if isinstance(review, str))


def clean_text_for_cloud(text):
    """
    Preparing the text for the word cloud
    :param data: parks dataset
    :return: text without conjunctions, prepositions and stopwords
    """
    stop_words = set(stopwords.words('english'))
    conjunctions_prepositions = r'\b(and|but|or|nor|for|yet|so|'
    conjunctions_prepositions += r"at|by|for|in|of|on|to|with|from|into|through|under|over|above|below|even|one|u|s|n't|)\b"
    pattern = re.compile(conjunctions_prepositions, flags=re.IGNORECASE)
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(cleaned_words)
    cleaned_text = pattern.sub('', cleaned_text)
    return cleaned_text


text_for_cloud = clean_text_for_cloud(reviews_text)

# Word cloud for all reviews of all parks
wordcloud = WordCloud(width=800, height=800,
                      background_color='white').generate(text_for_cloud)
plt.figure(figsize=(10, 5), facecolor=None)
plt.imshow(wordcloud,  interpolation='bilinear')
plt.axis("off")
plt.title("Reviews Word Cloud for All Parks Together")
plt.tight_layout(pad=0)
plt.show()


# # Creating a word cloud by parks
# ################################

def creat_words_cloud(category, text_data):
    """
    A general function to create a word cloud for each park separately
    :param data: parks dataset
    :return: a word cloud for the park it receives
    """
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Reviews Word Cloud for: {category}')
    plt.axis('off')
    plt.show()


for category in reviews_data["Branch"].unique():
    category_data = reviews_data[reviews_data['Branch']
                                 == category]['Review_Text']
    category_data = ' '.join(category_data)
    cloud_text = clean_text_for_cloud(category_data)
    creat_words_cloud(category, cloud_text)


# Sentiment analysis
####################

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity


reviews_data['Sentiment'] = reviews_data['Review_Text'].apply(
    analyze_sentiment)
sentiment_Stat = reviews_data['Sentiment'].describe()

# 1= extremely positive response, -1 extremely negative response
print(sentiment_Stat)


# Reviewer sentiment by Rating
##############################


# Normalization of the data in order to compare with the ratings
reviews_data['normalized_sentiment'] = (reviews_data["Sentiment"]+1)*2.5


def scatter_plot(x_label, x_values, y_label, y_values, title):
    """
    A general function to create a scatter plot
    :param data: parks dataset
    """
    y_vals = np.unique(y_values)
    mean_x = []
    mean_y = []
    for val in y_vals:
        index = (y_values == val)
        mean_x.append(np.mean(x_values[index]))
        mean_y.append(val)

    plt.scatter(x_values, y_values, color='c', label='Data Points')
    plt.scatter(mean_x, mean_y, color='crimson',
                marker='o', s=100, label='Mean')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()


def sort_by(data, sort_by):
    sorted_data = data.sort_values(by=sort_by)
    return sorted_data


sort_by(reviews_data, "normalized_sentiment")
scatter_plot("normalized_sentiment",
             reviews_data['normalized_sentiment'], "Rating", reviews_data["Rating"], "rating by normalized sentiment    ")

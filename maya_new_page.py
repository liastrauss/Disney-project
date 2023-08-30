import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
# nltk.download('punkt')-just if it is not uptodate
# nltk.download('stopwords')- just if it is not uptodate

# Data cleaning
###############

reviews_data = pd.read_csv('DisneylandReviews.csv',
                           encoding="cp1252", na_values=['missing'])  # Using the correct encoding is crucial to ensure that the characters in your text file are correctly interpreted

# here we delete 538~ that containe "missing" string
reviews_data = reviews_data.dropna(how='any')


# Sentiment analysis
####################

def analyze_sentiment(text):
    # Create a TextBlob object for the text
    blob = TextBlob(text)

    # Get the polarity (sentiment) of the text
    polarity = blob.sentiment.polarity

    return polarity


column_name = 'Review_Text'
reviews_data['Sentiment'] = reviews_data[column_name].apply(analyze_sentiment)
# reviews_data['normalized_sentiment'] = (reviews_data["Sentiment"]+1)*2.5
# 1= extremely positive response, -1 extremely negative response
sentiment_Stat = reviews_data['Sentiment'].describe()

# print(
#     f"the normalized sentiment= {reviews_data['Sentiment'], reviews_data['normalized_sentiment'].head}")
# print(sentiment_Stat)


# word cloud
############

# Combining all reviews
reviews_text = ' '.join(
    str(review) for review in reviews_data[column_name] if isinstance(review, str))


# Deletionconjunctions and prepositions
def clean_text_for_cloud(text):
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
print(text_for_cloud[:300])


# Word cloud for all reviews of all parks
# wordcloud = WordCloud(width=800, height=800,
#                       background_color='white').generate(text_for_cloud)
# plt.figure(figsize=(10, 5), facecolor=None)
# plt.imshow(wordcloud,  interpolation='bilinear')
# plt.axis("off")
# plt.title("Reviews Word Cloud for All Parks Together")
# plt.tight_layout(pad=0)
# plt.show()


# # Creating a word cloud by parks
# ################################

def creat_words_cloud(category, text_data):
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Reviews Word Cloud for: {category}')
    plt.axis('off')
    plt.show()


# for category in reviews_data["Branch"].unique():
#     category_data = reviews_data[reviews_data['Branch']
#                                  == category]['Review_Text']
#     category_data = ' '.join(category_data)
#     cloud_text = clean_text_for_cloud(category_data)
#     creat_words_cloud(category, cloud_text)


# # formatting the date column:
###############################

reviews_data['Year_Month'] = pd.to_datetime(
    reviews_data['Year_Month'], format='%Y-%m')

# creating a year and mounth column
reviews_data['year'] = pd.to_datetime(reviews_data['Year_Month']).dt.year
reviews_data['month'] = pd.to_datetime(reviews_data['Year_Month']).dt.month


# Reviewer_Location& Rating,  Reviewer_Location&Sentiment
#########################################################

#  Creating scatter plot
def scatter_plot(x_label, x_values, y_label, y_values, title):

    y_vals = np.unique(y_values)
    mean_x = []
    mean_y = []
    for val in y_vals:
        index = (y_values == val)
        mean_x.append(np.mean(x_values[index]))
        mean_y.append(val)

    # Plot markers for the means
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


# sort_by(reviews_data, "normalized_sentiment")
# scatter_plot("normalized_sentiment",
#              reviews_data['normalized_sentiment'], "Rating", reviews_data["Rating"], "rating by normalized sentiment    ")
# scatter_plot("Reviewer_Location", mean_ratings["Reviewer_Location"], "Rating",
#              mean_ratings["Rating"], "Rating by Reviewer_Location")


# # sorted_data = sort_by(reviews_data, "Rating")
# sorted_data = reviews_data.groupby("Reviewer_Location")[
#     "Sentiment"].mean()
# sorted_data = sorted_data.reset_index()
# sorted_data = sort_by(sorted_data, "Sentiment")
# top_ratings = sorted_data.head(25)

# #bar plot
# plt.figure(figsize=(8, 8))
# plt.barh(top_ratings["Reviewer_Location"],
#          top_ratings["Sentiment"], color="coral")
# plt.xlabel('Mean Sentiment')
# plt.ylabel('Reviewer Location')
# plt.title('Bottom Mean Sentiment by Reviewer Location')
# # plt.gca().invert_yaxis()
# plt.yticks(rotation=45)
# plt.show()

# get_top_words- אולי אפששר להשתמש כדי ליעל את הפונקציות שלמעלה
# #############
def get_top_words(texts, top_n=10):
    words = ' '.join(texts).lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    word_count = Counter(words)
    return word_count.most_common(top_n)

# top_words_per_location = reviews_data.groupby(
#     'Reviewer_Location')['Review_Text'].apply(lambda x: get_top_words(x, top_n=10))


# Ranking by parks- !!!!!!!!
###################
def Ranking_by_parks():
    plt.figure(figsize=(8, 8))
    plt.title('Ranking by parks')
    sns.boxplot(x="Branch", y="Rating", data=reviews_data)
    plt.show()

# Ranking_by_parks()


# Create a bar plot
def bar_plot(x_value, y_value, data):
    plt.bar(data[x_value], data[y_value])
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.title(f"{y_value} by {x_value}")
    plt.legend()
    plt.show()

# bar plot of The 20 countries with the most reviews'
####################################################


def bar_plot_catagorys(column, x_lable, y_lable, title):
    category_counts = reviews_data[column].value_counts()
    category_counts.sort_by(category_counts, "year")
    category_counts = category_counts.head(20)
    category_counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.xticks(rotation=30)
    plt.show()


# bar_plot_catagorys("Reviewer_Location", 'Countries',
#                    'Number of reviews', 'The 20 countries with the most reviews')
# sort_by(reviews_data, "year")
# bar_plot_catagorys("year", 'year',
#                    'Number of reviews', 'number of reviews per year')

# bar plot of The number of reviews per year'
####################################################

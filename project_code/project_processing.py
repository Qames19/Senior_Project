# import modules necessary for data analysis
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import datetime as dt
import string
import yfinance as yf
import re
from spellchecker import SpellChecker

# read Company_Tweet.csv and Tweet.csv into dataframes
company_tweet = pd.read_csv('Company_Tweet.csv')
tweet = pd.read_csv('Tweet.csv', encoding='utf-8')

# Company.csv will not be used in my analysis
# company = pd.read_csv('D:\\Data_Senior_Project\\Company.csv')

# merge dataframes
tweets = pd.merge(tweet, company_tweet, on='tweet_id', how='inner')

# convert post_date to absolute time from relative time & create date column to represent date without timestamp
tweets['post_date'] = pd.to_datetime(tweets['post_date'], unit='s')
tweets['date'] = pd.to_datetime(tweets['post_date'].apply(lambda date: date.date()))

# remove duplicate tweets
tweets = tweets.drop_duplicates(subset='body')

# create a dataframe for stock prices over the period of the study for all companies in the study
beginning_date = dt.date(2014, 12, 30)
ending_date = dt.date(2020, 1, 3)
stocks = yf.download('AAPL AMZN GOOG GOOGL MSFT TSLA', start=beginning_date, end=ending_date)
stocks = stocks.drop(['Close', 'Volume', 'Open', 'High', 'Low'], axis=1)
stocks.columns = stocks.columns.droplevel()

# remove tweets from days where the stock market is closed
valid_days = [x for x in stocks.index.values]
tweets = tweets[tweets['date'].isin(valid_days)]

# use stock info to create column with change in adj. close from business day before tweet to business day after tweet
stocks['AAPL2'] = stocks['AAPL'].shift(periods=-1) - stocks['AAPL'].shift(periods=1)
stocks['AMZN2'] = stocks['AMZN'].shift(periods=-1) - stocks['AMZN'].shift(periods=1)
stocks['GOOG2'] = stocks['GOOG'].shift(periods=-1) - stocks['GOOG'].shift(periods=1)
stocks['GOOGL2'] = stocks['GOOGL'].shift(periods=-1) - stocks['GOOGL'].shift(periods=1)
stocks['MSFT2'] = stocks['MSFT'].shift(periods=-1) - stocks['MSFT'].shift(periods=1)
stocks['TSLA2'] = stocks['TSLA'].shift(periods=-1) - stocks['TSLA'].shift(periods=1)

# drop original stock columns
stocks = stocks.drop(['AAPL', 'AMZN', 'GOOG', 'GOOGL', 'MSFT', 'TSLA'], axis=1)

# ensure tweets['date'] and stocks.index are the same data type
tweets['date'] = pd.to_datetime(tweets['date'], format='%Y-%m-%d')
stocks.index = pd.to_datetime(stocks.index, format="%Y-%m-%d")

# merge tweets and stocks
tweets = pd.merge(tweets, stocks, left_on='date', right_index=True)

# match the change in price to the correct ticker symbol
tweets.at[tweets.ticker_symbol[tweets.ticker_symbol == 'AAPL'].index, 'price_diff'] = tweets['AAPL2']
tweets.at[tweets.ticker_symbol[tweets.ticker_symbol == 'AMZN'].index, 'price_diff'] = tweets['AMZN2']
tweets.at[tweets.ticker_symbol[tweets.ticker_symbol == 'GOOG'].index, 'price_diff'] = tweets['GOOG2']
tweets.at[tweets.ticker_symbol[tweets.ticker_symbol == 'GOOGL'].index, 'price_diff'] = tweets['GOOGL2']
tweets.at[tweets.ticker_symbol[tweets.ticker_symbol == 'MSFT'].index, 'price_diff'] = tweets['MSFT2']
tweets.at[tweets.ticker_symbol[tweets.ticker_symbol == 'TSLA'].index, 'price_diff'] = tweets['TSLA2']

# drop the columns from stocks
tweets = tweets.drop(['AAPL2', 'AMZN2', 'GOOG2', 'GOOGL2', 'MSFT2', 'TSLA2'], axis=1)

# change price_diff from quantitative to qualitative
conditions = [
    (tweets['price_diff'] > 0),
    (tweets['price_diff'] == 0),
    (tweets['price_diff'] < 0)
]
values = ['increase', 'no change', 'decrease']
tweets['price_change'] = np.select(conditions, values)

# only 2000 values have 'no change'.  These entries will be removed to improve testing
tweets = tweets[tweets['price_change'] != 'no change']

# equalize power of likes, retweets, and comments - categorize interaction level
conditions = [
    (tweets['comment_num'] > 0) | (tweets['retweet_num'] > 0) | (tweets['like_num'] > 0),
    (tweets['comment_num'] == 0) & (tweets['retweet_num']) == 0 & (tweets['like_num'] == 0)
]
values = [True, False]

tweets['at_least_one'] = np.select(conditions, values)

at_least_one = tweets[tweets['at_least_one'] == True]

comment_avg_no_zeroes = at_least_one.describe().loc[['mean'], ['comment_num']]
comment_avg_no_zeroes = float(comment_avg_no_zeroes['comment_num'])

retweets_avg_no_zeroes = at_least_one.describe().loc[['mean'], ['retweet_num']]
retweets_avg_no_zeroes = float(retweets_avg_no_zeroes['retweet_num'])

like_avg_no_zeroes = at_least_one.describe().loc[['mean'], ['like_num']]
like_avg_no_zeroes = float(like_avg_no_zeroes['like_num'])

likes_to_comments = like_avg_no_zeroes / comment_avg_no_zeroes
likes_to_retweets = like_avg_no_zeroes / retweets_avg_no_zeroes

tweets["combined_int"] = (tweets["comment_num"] * likes_to_comments) + \
                         (tweets["retweet_num"] * likes_to_retweets) + tweets["like_num"]

# set low interaction > 0 < (mean-1sd), high interaction as > (mean + 1sd), medium as (mean +/- 1sd)
conditions = [
    (tweets['combined_int'] > 11),
    (tweets['combined_int'] > 2) & (tweets['combined_int'] <= 11),
    (tweets['combined_int'] > 0) & (tweets['combined_int'] <= 3),
    (tweets['combined_int'] == 0)
]
values = ['high', 'medium', 'low', 'none']

tweets['interaction_level'] = np.select(conditions, values)

tweets.drop(['combined_int', 'at_least_one'], axis=1, inplace=True)


# function to add meta-features to dataset
def create_meta(train_data):
    train_data['original_text'] = train_data['body']

    # Number of words in the text #
    train_data["num_words"] = train_data["body"].apply(lambda x: len(str(x).split()))

    # Number of unique words in the text #
    train_data["num_unique_words"] = train_data["body"].apply(lambda x: len(set(str(x).split())))

    # Number of characters in the text #
    train_data["num_chars"] = train_data["body"].apply(lambda x: len(str(x)))

    # Number of punctuations in the text #  no results returned. function removed
    train_data["num_punctuations"] = train_data['body'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))

    # Number of words upper in the text
    train_data["num_words_upper"] = train_data["body"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]))

    # Number of title case words in the text #
    train_data["num_words_title"] = train_data["body"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))

    # Average length of the words in the text #
    train_data["mean_word_len"] = train_data["body"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # Number of special characters
    train_data['num_special_char'] = train_data['body'].str.findall(r'[^a-zA-Z0-9 ]').str.len()

    # Number of numerics
    train_data['num_numerics'] = train_data['body'].apply(lambda x: sum(c.isdigit() for c in x))

    # Number of Upper case in text
    train_data['num_uppercase'] = train_data['body'].apply(
        lambda x: len([nu for nu in str(x).split() if nu.isupper()]))

    # Number of Lower case in text
    train_data['num_lowercase'] = train_data['body'].apply(
        lambda x: len([nl for nl in str(x).split() if nl.islower()]))

    # Remove special characters
    spec_chars = [
        "!", '"', "#", "%", "&", "'", "(", ")",
        "*", "+", ",", "-", ".", "/", ":", ";", "<",
        "=", ">", "?", "@", "[", "\\", "]", "^", "_",
        "`", "{", "|", "}", "~", "–"
    ]
    for char in spec_chars:
        train_data['body'] = train_data['body'].str.replace(char, ' ')
    train_data['body'] = train_data['body'].str.split().str.join(" ")

    # create polarity meta-feature
    train_data['polarity'] = train_data.body.apply(lambda x: TextBlob(str(x)).sentiment.polarity)


create_meta(tweets)

# change polarity to sentiment.
conditions = [
    (tweets['polarity'] <= -0.5),
    (tweets['polarity'] > -0.5) & (tweets['polarity'] < 0.5),
    (tweets['polarity'] >= 0.5)
]
values = ['negative', 'neutral', 'positive']
tweets['sentiment'] = np.select(conditions, values)

# save modified data just in case
tweets.to_csv('outputs//tweet2.csv', index=False)

# divide data by company
aapl = tweets[tweets['ticker_symbol'] == 'AAPL']
amzn = tweets[tweets['ticker_symbol'] == 'AMZN']
goog = tweets[tweets['ticker_symbol'] == 'GOOG']
googl = tweets[tweets['ticker_symbol'] == 'GOOGL']
msft = tweets[tweets['ticker_symbol'] == 'MSFT']
tsla = tweets[tweets['ticker_symbol'] == 'TSLA']

# create price based training group
aapl_price_train = aapl.groupby('price_change').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
amzn_price_train = amzn.groupby('price_change').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
goog_price_train = goog.groupby('price_change').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
googl_price_train = googl.groupby('price_change').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
msft_price_train = msft.groupby('price_change').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
tsla_price_train = tsla.groupby('price_change').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)

# remove the columns used to derive the price_change column from the price based training groups
aapl_price_train.drop(['price_diff'], axis=1)
amzn_price_train.drop(['price_diff'], axis=1)
goog_price_train.drop(['price_diff'], axis=1)
googl_price_train.drop(['price_diff'], axis=1)
msft_price_train.drop(['price_diff'], axis=1)
tsla_price_train.drop(['price_diff'], axis=1)

# create interaction based training group
aapl_interaction_train = aapl.groupby('interaction_level').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
amzn_interaction_train = amzn.groupby('interaction_level').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
goog_interaction_train = goog.groupby('interaction_level').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
googl_interaction_train = googl.groupby('interaction_level').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
msft_interaction_train = msft.groupby('interaction_level').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
tsla_interaction_train = tsla.groupby('interaction_level').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)

# remove the columns used to derive the interaction_level column from the interaction based training groups
aapl_interaction_train.drop(['retweet_num', 'like_num', 'comment_num'], axis=1)
amzn_interaction_train.drop(['retweet_num', 'like_num', 'comment_num'], axis=1)
goog_interaction_train.drop(['retweet_num', 'like_num', 'comment_num'], axis=1)
googl_interaction_train.drop(['retweet_num', 'like_num', 'comment_num'], axis=1)
msft_interaction_train.drop(['retweet_num', 'like_num', 'comment_num'], axis=1)
tsla_interaction_train.drop(['retweet_num', 'like_num', 'comment_num'], axis=1)

# create sentiment based training group
aapl_sentiment_train = aapl.groupby('sentiment').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
amzn_sentiment_train = amzn.groupby('sentiment').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
goog_sentiment_train = goog.groupby('sentiment').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)
googl_sentiment_train = googl.groupby('sentiment').apply(lambda x: x.sample(n=979)).reset_index(drop=True)
msft_sentiment_train = msft.groupby('sentiment').apply(lambda x: x.sample(n=709)).reset_index(drop=True)
tsla_sentiment_train = tsla.groupby('sentiment').apply(lambda x: x.sample(n=1500)).reset_index(drop=True)

# remove the columns used to derive the sentiment column from the sentiment based training groups
aapl_sentiment_train.drop(['polarity'], axis=1)
amzn_sentiment_train.drop(['polarity'], axis=1)
goog_sentiment_train.drop(['polarity'], axis=1)
googl_sentiment_train.drop(['polarity'], axis=1)
msft_sentiment_train.drop(['polarity'], axis=1)
tsla_sentiment_train.drop(['polarity'], axis=1)

def long_meta(train_data):
    def count_explicit_content(train_data):
        count = 0
        file = open('swearWords.txt', 'r')
        swearwords = file.read()
        swearwords = swearwords.split('\n')
        file.close()
        # curr = 0
        for entry in train_data['body']:
            words = re.findall(r'\w+', entry)
            badword = [phrase for phrase in swearwords if (phrase in words)]

            if badword:
                count += 1
                train_data.at[train_data.body[train_data.body == entry].index, 'isExplicit'] = 1
            else:
                train_data.at[train_data.body[train_data.body == entry].index, 'isExplicit'] = 0
        print(count)
        return train_data

    def count_typo(sent):
        return len(find_typo(sent))

    def find_typo(sent):
        spell = SpellChecker()
        words = sent.split()
        words = spell.unknown(words)
        return words

    count_explicit_content(train_data)

    train_data['typo'] = train_data.body.apply(count_typo)


# add time-intensive features to _price_train
long_meta(aapl_price_train)
long_meta(amzn_price_train)
long_meta(goog_price_train)
long_meta(googl_price_train)
long_meta(msft_price_train)
long_meta(tsla_price_train)

# add time-intensive features to _interaction_train
long_meta(aapl_interaction_train)
long_meta(amzn_interaction_train)
long_meta(goog_interaction_train)
long_meta(googl_interaction_train)
long_meta(msft_interaction_train)
long_meta(tsla_interaction_train)

# add time-intensive features to _sentiment_train
long_meta(aapl_sentiment_train)
long_meta(amzn_sentiment_train)
long_meta(goog_sentiment_train)
long_meta(googl_sentiment_train)
long_meta(msft_sentiment_train)
long_meta(tsla_sentiment_train)


def create_tfdf(df_sample, classifier_name, features):
    name = "sample_" + classifier_name

    print(df_sample.head(10))

    print("Calculating top term frequency by inverse document frequency matrix...")
    vectorizer = CountVectorizer(max_df=0.95, max_features=features, binary=False)
    counts = vectorizer.fit_transform(df_sample.body)

    # Transforms the data into a bag of words
    bag_of_words = vectorizer.transform(df_sample.body)

    # find maximum value for each of the features over dataset:
    max_value = bag_of_words.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()

    # get feature names
    feature_names = np.array(vectorizer.get_feature_names())
    print("Features with lowest tfidf:\n{}".format(
        feature_names[sorted_by_tfidf[:features]]))
    print("\nFeatures with highest tfidf: \n{}".format(
        feature_names[sorted_by_tfidf[-features:]]))

    # find maximum value for each of the features over all of dataset:
    max_val = bag_of_words.max(axis=0).toarray().ravel()

    # sort weights from smallest to biggest and extract their indices
    sort_by_tfidf = max_val.argsort()
    print("Features with lowest tfidf:\n{}".format(
        feature_names[sort_by_tfidf[:features]]))
    print("\nFeatures with highest tfidf: \n{}".format(
        feature_names[sort_by_tfidf[-features:]]))

    tdf_count_vectorizer = pd.DataFrame(counts.toarray(), columns=vectorizer.get_feature_names())
    name1 = "outputs//" + name + "_TFIDF" + str(features) + ".csv"
    tdf_count_vectorizer.to_csv(name1)
    print(tdf_count_vectorizer)

    dfnew = pd.read_csv(name1)
    combined_df = pd.concat([df_sample, dfnew], ignore_index=False, sort=False, axis=1, join="inner")
    name2 = "outputs//combined_" + name + "_TFIDF" + str(features) + ".csv"
    combined_df.to_csv(name2)
    text = " ".join(df_sample.body)


# add time-intensive features to _price_train
create_tfdf(aapl_price_train, 'aapl_price', 100)
create_tfdf(amzn_price_train, 'amzn_price', 100)
create_tfdf(goog_price_train, 'goog_price', 100)
create_tfdf(googl_price_train, 'googl_price', 100)
create_tfdf(msft_price_train, 'msft_price', 100)
create_tfdf(tsla_price_train, 'tsla_price', 100)

# add time-intensive features to _interaction_train
create_tfdf(aapl_interaction_train, 'aapl_interaction', 100)
create_tfdf(amzn_interaction_train, 'amzn_interaction', 100)
create_tfdf(goog_interaction_train, 'goog_interaction', 100)
create_tfdf(googl_interaction_train, 'googl_interaction', 100)
create_tfdf(msft_interaction_train, 'msft_interaction', 100)
create_tfdf(tsla_interaction_train, 'tsla_interaction', 100)

# add time-intensive features to _sentiment_train
create_tfdf(aapl_sentiment_train, 'aapl_sentiment', 100)
create_tfdf(amzn_sentiment_train, 'amzn_sentiment', 100)
create_tfdf(goog_sentiment_train, 'goog_sentiment', 100)
create_tfdf(googl_sentiment_train, 'googl_sentiment', 100)
create_tfdf(msft_sentiment_train, 'msft_sentiment', 100)
create_tfdf(tsla_sentiment_train, 'tsla_sentiment', 100)

# aapl_price_train.to_csv('outputs//aapl_price_train.csv', index=False)
# amzn_price_train.to_csv('outputs//amzn_price_train.csv', index=False)
# goog_price_train.to_csv('outputs//goog_price_train.csv', index=False)
# googl_price_train.to_csv('outputs//googl_price_train.csv', index=False)
# msft_price_train.to_csv('outputs//msft_price_train.csv', index=False)
# tsla_price_train.to_csv('outputs//tsla_price_train.csv', index=False)
#
# aapl_interaction_train.to_csv('outputs//aapl_interaction_train.csv', index=False)
# amzn_interaction_train.to_csv('outputs//amzn_interaction_train.csv', index=False)
# goog_interaction_train.to_csv('outputs//goog_interaction_train.csv', index=False)
# googl_interaction_train.to_csv('outputs//googl_interaction_train.csv', index=False)
# msft_interaction_train.to_csv('outputs//msft_interaction_train.csv', index=False)
# tsla_interaction_train.to_csv('outputs//tsla_interaction_train.csv', index=False)
#
# aapl_sentiment_train.to_csv('outputs//aapl_sentiment_train.csv', index=False)
# amzn_sentiment_train.to_csv('outputs//amzn_sentiment_train.csv', index=False)
# goog_sentiment_train.to_csv('outputs//goog_sentiment_train.csv', index=False)
# googl_sentiment_train.to_csv('outputs//googl_sentiment_train.csv', index=False)
# msft_sentiment_train.to_csv('outputs//msft_sentiment_train.csv', index=False)
# tsla_sentiment_train.to_csv('outputs//tsla_sentiment_train.csv', index=False)

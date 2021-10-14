import pandas as pd
import numpy as np
import nltk
import string
import re
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from spellchecker import SpellChecker
from textblob import TextBlob
from nltk.util import ngrams
from collections import Counter
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer


""" Create Set of Stopwords """

stop_words = nltk.corpus.stopwords.words('english')
my_stop_words = {'http', 'com', 'bit', 'ly', 'utm', 'index', 'html', 'bi', 'twitter', 'source', 'medium', 'iwm', 'qqq'
               'stks', 'co', 'xref', 'utm_source', 'utm_medium', 'https', 'utm_campaign', '1126416â', 'php', 'js',
               'dlvr', 'htm', 'owler', 'us', 'read', 'tweet', 'spy'}

my_stop_words |= set(stop_words)


""" Function that creates bag of words and appends to the current data """


def create_tfdf(df_sample_2000, rows, features):
    name = "sample_" + str(rows)
    halfRows=int(rows/2)
    df_sample_2000 = df.groupby('3-interactions').apply(lambda x: x.sample(n=halfRows)).reset_index(drop=True)

    df_sample_2000 = create_meta(df_sample_2000)

    create_ngrams(df_sample_2000)

    print(df_sample_2000.head(10))

    print("Calculating top term frequency by inverse document frequency matrix...")
    vectorizer = CountVectorizer(max_df=0.95, max_features=features, binary=False)
    counts = vectorizer.fit_transform(df_sample_2000.body)

    # Transforms the data into a bag of words
    bag_of_words = vectorizer.transform(df_sample_2000.body)

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
    name1 = name + "TFIDF" + str(features) + ".csv"
    tdf_count_vectorizer.to_csv(name1)
    print(tdf_count_vectorizer)

    dfnew = pd.read_csv(name1)
    combined_df = pd.concat([df_sample_2000, dfnew], ignore_index=False, sort=False, axis=1, join="inner")
    name2 = "combined" + name1
    combined_df.to_csv(name2)
    text = " ".join(df_sample_2000.body)

    # Create the wordcloud object
    wordcloud = WordCloud(width=1024, height=1024, margin=0, stopwords=my_stop_words).generate(text)

    # Display the generated image:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    #plt.show()
    plt.savefig("project_wordcloud.png")


def create_meta(train_data):
    train_data['original_text'] = train_data['body']

    # Number of words in the text #
    train_data["num_words"] = train_data["body"].apply(lambda x: len(str(x).split()))

    # Number of unique words in the text #
    train_data["num_unique_words"] = train_data["body"].apply(lambda x: len(set(str(x).split())))

    # Number of characters in the text #
    train_data["num_chars"] = train_data["body"].apply(lambda x: len(str(x)))

    # Number of punctuations in the text #
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

    count_explicit_content(train_data)

    train_data['typo'] = train_data.body.apply(count_typo)

    detect_sell_buy(train_data)

    train_data['polarity'] = train_data.body.apply(detect_polarity)

    return train_data


def create_ngrams(train_data):
    df_unpopular, df_popular = train_data.loc[train_data['3-interactions'] == False], \
                               train_data[train_data['3-interactions'] == True]
    df_unpopular = append_col(df_unpopular)
    df_popular = append_col(df_popular)

    tokens = []

    fdist = FreqDist(tokens)
    print(fdist)

    popular_claim = create_corpus(df_popular)
    unpopular_claim = create_corpus(df_unpopular)

    text_trends(popular_claim, "Popular Claims")
    text_trends(unpopular_claim, "Unpopular Claims")


#   *** Begin helper functions for create_meta function ***

# Adding feature to determine if the words buy or sell are in the tweet
def detect_sell_buy(train_data):
    count_sell = 0
    count_buy = 0
    for entry in train_data['body']:
        words = re.findall(r'\w+', entry)
        if 'sell' in words:
            count_sell += 1
            train_data.at[train_data.body[train_data.body == entry].index, 'has_Sell'] = 1
        else:
            train_data.at[train_data.body[train_data.body == entry].index, 'has_Sell'] = 0
        if 'buy' in words:
            count_buy += 1
            train_data.at[train_data.body[train_data.body == entry].index, 'has_Buy'] = 1
        else:
            train_data.at[train_data.body[train_data.body == entry].index, 'has_Buy'] = 0
    return train_data

# Explicit feature

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
    return train_data


def get_word_list(text):
    return re.sub("[^\w]", " ", text.upper()).split()


def count_typo(sent):
    return len(find_typo(sent))


def find_typo(sent):
    spell = SpellChecker()
    words = sent.split()
    words = spell.unknown(words)
    return words


#     Adding feature to judge sentiment by polarity
#     I followed the classroom instruction and found every polarity to be the same (input error).
#     I decided that it seemed to be applying the polarity to the entire dataframe instead of to each entry in the df
#     I searched for solutions and found the following solution at
#     https://towardsdatascience.com/having-fun-with-textblob-7e9eed783d3f
#     I modified the code to work with my data 


def detect_polarity(text):
    return TextBlob(text).sentiment.polarity


#  *** End of create_meta helper functions ***

#  *** Begin n-gram helper functions ***


# First function, cleanReviews, cleans the reviews by stripping punctuation and whitespace, converting to lowercase, 
# and removing stopwords (you'll see I added some domain specific stopwords).


def clean_reviews(documents):
    cleaned_reviews = []
    for document in documents:
        s = re.sub(r'[^a-zA-Z0-9\s]', '', document)
        s = re.sub('\s+', ' ', s)
        s = str(s).lower()
        tokens = [token for token in s.split(" ") if token != ""]
        tokens = [word for word in tokens if word not in my_stop_words]
        review = ' '.join(tokens)
        cleaned_reviews.append(review)
    return cleaned_reviews


# Second function, documentNgrams, takes the clean reviews and computes [n-grams](https://en.wikipedia.org/wiki/N-gram)
# of any size specified by the user. This function then packages the 15 most common n-grams into a pandas dataframe.


def document_ngrams(documents, size):
    ngrams_all = []
    for document in documents:
        tokens = document.split()
        if len(tokens) <= size:
            continue
        else:
            output = list(ngrams(tokens, size))
        for ngram in output:
            ngrams_all.append(" ".join(ngram))
    cnt_ngram = Counter()
    for word in ngrams_all:
        cnt_ngram[word] += 1
    train_data = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    train_data = train_data.rename(columns={'index': 'words', 0: 'count'})
    train_data = train_data.sort_values(by='count', ascending=False)
    train_data = train_data.head(15)
    train_data = train_data.sort_values(by='count')
    return train_data


# The third function, plotNgramsChar, computes n-grams of size 1, 2, and 3 by characters in each dataframe, it then 
# creates three horizontal bar charts for each n-gram size.

def document_ngrams_ch(documents, size):
    ngrams_all = []
    for document in documents:
        tokens = document.split()
        for characters in tokens:
            if len(characters) <= size:
                continue
            else:
                output = list(ngrams(characters, size))
            for ngram in output:
                ngrams_all.append(" ".join(ngram))
    cnt_ngram = Counter()
    for word in ngrams_all:
        cnt_ngram[word] += 1
    train_data = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    train_data = train_data.rename(columns={'index': 'words', 0: 'count'})
    train_data = train_data.sort_values(by='count', ascending=False)
    train_data = train_data.head(15)
    train_data = train_data.sort_values(by='count')
    return train_data


# The fourth function, plotNgrams, computes n-grams of size 1, 2, and 3. Using each of the n-gram dataframes, it then 
# creates three horizontal bar charts for each n-gram size.

def plot_ngrams(documents, type):
    unigrams = document_ngrams(documents, 1)
    bigrams = document_ngrams(documents, 2)
    trigrams = document_ngrams(documents, 3)
    fourgrams = document_ngrams(documents, 4)

    # Set plot figure size
    fig = plt.figure(figsize=(16, 7))
    plt.subplots_adjust(wspace=.5)

    ax = fig.add_subplot(141)
    ax.barh(np.arange(len(unigrams['words'])), unigrams['count'], align='center', alpha=.5)
    title = 'Unigrams ' + type
    ax.set_title(title)
    plt.yticks(np.arange(len(unigrams['words'])), unigrams['words'])
    plt.xlabel('Count')

    ax2 = fig.add_subplot(142)
    ax2.barh(np.arange(len(bigrams['words'])), bigrams['count'], align='center', alpha=.5)
    title = 'Bigrams ' + type
    ax2.set_title(title)
    plt.yticks(np.arange(len(bigrams['words'])), bigrams['words'])
    plt.xlabel('Count')

    ax3 = fig.add_subplot(143)
    ax3.barh(np.arange(len(trigrams['words'])), trigrams['count'], align='center', alpha=.5)
    title = 'Trigrams ' + type
    ax3.set_title(title)
    plt.yticks(np.arange(len(trigrams['words'])), trigrams['words'])
    plt.xlabel('Count')

    ax4 = fig.add_subplot(144)
    ax4.barh(np.arange(len(fourgrams['words'])), fourgrams['count'], align='center', alpha=.5)
    title = 'Fourgrams ' + type
    ax4.set_title(title)
    plt.yticks(np.arange(len(fourgrams['words'])), fourgrams['words'])
    plt.xlabel('Count')

    fig.tight_layout()
    #plt.show()
    plt.savefig(type + 'n-grams.png')


# The fifth function, plotNgrams, computes n-grams of size 1, 2, 3, and 4. Using each of the n-gram dataframes by 
# character, it then creates three horizontal bar charts for each n-gram size.

def plot_ngrams_ch(documents, type):
    unigrams_ch = document_ngrams_ch(documents, 1)
    bigrams_ch = document_ngrams_ch(documents, 2)
    trigrams_ch = document_ngrams_ch(documents, 3)
    fourgrams_ch = document_ngrams_ch(documents, 4)

    # Set plot figure size
    fig = plt.figure(figsize=(16, 7))
    plt.subplots_adjust(wspace=.5)

    ax = fig.add_subplot(141)
    ax.barh(np.arange(len(unigrams_ch['words'])), unigrams_ch['count'], align='center', alpha=.5)
    title = 'Unigrams by Character' + type
    ax.set_title(title)
    plt.yticks(np.arange(len(unigrams_ch['words'])), unigrams_ch['words'])
    plt.xlabel('Count')

    ax2 = fig.add_subplot(142)
    ax2.barh(np.arange(len(bigrams_ch['words'])), bigrams_ch['count'], align='center', alpha=.5)
    title = 'Bigrams ' + type
    ax2.set_title(title)
    plt.yticks(np.arange(len(bigrams_ch['words'])), bigrams_ch['words'])
    plt.xlabel('Count')

    ax3 = fig.add_subplot(143)
    ax3.barh(np.arange(len(trigrams_ch['words'])), trigrams_ch['count'], align='center', alpha=.5)
    title = 'Trigrams ' + type
    ax3.set_title(title)
    plt.yticks(np.arange(len(trigrams_ch['words'])), trigrams_ch['words'])
    plt.xlabel('Count')

    ax4 = fig.add_subplot(144)
    ax4.barh(np.arange(len(fourgrams_ch['words'])), fourgrams_ch['count'], align='center', alpha=.5)
    title = 'Fourgrams ' + type
    ax4.set_title(title)
    plt.yticks(np.arange(len(fourgrams_ch['words'])), fourgrams_ch['words'])
    plt.xlabel('Count')

    fig.tight_layout()
    plt.show()


def create_corpus(df_target):
    corpus = []
    for index, row in df_target.iterrows():
        text = row["body"]
        try:

            review = re.sub('[^a-zA-Z]', ' ', text)
            review = review.lower()
            review = review.split()
            review = ' '.join(review)
            corpus.append(review)
        except Exception as e:
            k = 3
            print(e)
    print(" a sample from the corpus")
    print(corpus[0:15])
    return corpus


# returns number of words in a sentence
def num_words(s):
    return len(s.split())


# adds a word to a column
def append_col(train_data):
    word_counts = []
    for index, row in train_data.iterrows():
        word_counts.append(num_words(row['body']))
    train_data['word count'] = word_counts
    return train_data


# The final function, textTrends, puts the previous functions together. Call this to start understanding what people
# are talking about.


def text_trends(documents, type):
    cleaned_reviews = clean_reviews(documents)
    plot_ngrams(cleaned_reviews, type)
    plot_ngrams_ch(cleaned_reviews, type)


if __name__ == '__main__':

    df = pd.read_csv('Tweet3.csv')
    print("Term Document matrix is: \n")

    # creating the corpus matrix
    print("Loading data...")

    corpus = []  # initializing a corpus variable
    print("before filtering: %d" % len(df.index))

    df = df.dropna(axis=0)
    print("after filtering: %d" % len(df.index))

    create_tfdf(df, 5000, 100)
    create_tfdf(df, 5000, 500)
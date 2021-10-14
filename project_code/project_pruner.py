import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
style.use("ggplot")

df = pd.read_csv('outputs/combined_sample_goog_sentiment_TFIDF100.csv', encoding='utf-8')
classifier_name = 'GOOG_sentiment'
column_name = 'sentiment'

# class names for interaction level
# class_names1 = ['low', 'none', 'medium', 'high']

# class names for price change
# class_names1 = ['increase', 'decrease']

# class names for sentiment
class_names1 = ['negative', 'neutral', 'positive']

print("Term Document matrix is: \n")
# creating the corpus matrix
print("Loading data...")
corpus = []  # initializing a corpus variable
print("before filtering: %d" % len(df.index))
df = df.dropna(axis=0)

# drop columns for interaction_level
# df = df.drop(['tweet_id', 'writer', 'post_date', 'body', 'date', 'original_text', 'ticker_symbol', 'like_num',
#               'comment_num', 'retweet_num', 'sentiment', 'price_change', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# drop columns for price_change
# df = df.drop(['tweet_id', 'writer', 'post_date', 'body', 'date', 'original_text', 'ticker_symbol', 'interaction_level',
#               'sentiment', 'price_diff', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# drop columns for sentiment
df = df.drop(['tweet_id', 'writer', 'post_date', 'body', 'date', 'original_text', 'ticker_symbol', 'interaction_level',
              'polarity', 'price_change', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)

print("after filtering: %d" % len(df.index))
train_data = pd.DataFrame.copy(df)

# create y column for training
y = train_data.loc[:, [column_name]].values

# create y column for regr training
y_regr = train_data.loc[:, [column_name]]
y_regr = pd.get_dummies(y_regr[column_name])

train_data = train_data.drop([column_name], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.20, random_state=42)
X_regr_train, X_regr_test, y_regr_train, y_regr_test = train_test_split(train_data, y_regr, test_size=0.20, random_state=42)
feature_names = list(train_data.columns.values)

''' finding the correct alpha for ccp.  Code is from StatQuest by Josh Starmer'''
DT = DecisionTreeClassifier(criterion='entropy')
path = DT.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    DT = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(DT, X_train, y_train, cv=10)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                              columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')

plt.show()

# ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.00147) & (alpha_results['alpha'] < 0.00175)]
# ideal_ccp_alpha = float(ideal_ccp_alpha['alpha'])
# print(ideal_ccp_alpha)

regr = DecisionTreeRegressor()
path = regr.cost_complexity_pruning_path(X_regr_train, y_regr_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    regr = DecisionTreeRegressor(criterion='mse', random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(regr, X_regr_train, y_regr_train, cv=10)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')

plt.show()

# print(alpha_results[(alpha_results['alpha'] > 0.0015) & (alpha_results['alpha'] < 0.00157)])
# ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.0015) & (alpha_results['alpha'] < 0.00157)]
# ideal_ccp_alpha = float(ideal_ccp_alpha['alpha'])
# print(ideal_ccp_alpha)

'''code to optimize RandomForestClassifier retrieved from YouTube user Kunaal Naik - https://youtu.be/c4mS7KaOIGY '''
# hyperparameters for parameter_grid
n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [2, 4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

# Parameter Grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

rf_Model = RandomForestClassifier()
rf_Grid = GridSearchCV(estimator=rf_Model, param_grid=random_grid, cv=3, verbose=2, n_jobs=4)

rf_Grid.fit(X_train, y_train)

print(rf_Grid.best_params_)

print(f'Train Accuracy: {rf_Grid.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {rf_Grid.score(X_test, y_test):.3f}')
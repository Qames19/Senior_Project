import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import graphviz
import pydotplus
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz  # remember to load the package
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
style.use("ggplot")

df = pd.read_csv('outputs/combined_sample_goog_sentiment_TFIDF100.csv', encoding='utf-8')
classifier_name = 'GOOG_sentiment_optimized'
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
print(y_regr.head())

train_data = train_data.drop([column_name], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.20, random_state=42)
feature_names = list(train_data.columns.values)


def decisionTreeWork():
    DT = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.0025)
    print("Decision Tree classifier")
    Y_pred_DT = DT.fit(X_train, y_train)
    predictions = Y_pred_DT.predict(X_test)

    cm0 = confusion_matrix(y_test, predictions)
    print(f"Confusion matrix for {classifier_name} Decision tree is: ")
    print(cm0)

    text_representation = tree.export_text(DT)
    with open(f"outputs/classifiers/{classifier_name}_decision_tree.log", "w") as fout:
        fout.write(text_representation)
    print(text_representation)
    fig = plt.figure(figsize=(40, 32))

    mytree = tree.plot_tree(DT,
                            feature_names=feature_names,
                            class_names=class_names1,
                            filled=True, fontsize=12)

    fig.savefig(f"outputs/classifiers/{classifier_name}_decision_tree.png")
    p, r, f1, s = precision_recall_fscore_support(y_true=y_test, y_pred=predictions, average='macro', warn_for=tuple())
    print("%.3f Precision\n%.3f Recall\n%.3f F1" % (p, r, f1))
    print(confusion_matrix(y_true=y_test, y_pred=predictions))
    print(classification_report(y_true=y_test, y_pred=predictions, zero_division=1))

    dot_data = tree.export_graphviz(DT, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names1,
                                    filled=True)

    # Draw graph

    graph = graphviz.Source(dot_data, format="png")
    print(graph.source)

    def save_png_clf(clf, filename, feature_names, class_names):
        dot_data = tree.export_graphviz(clf, feature_names=feature_names, class_names=class_names, filled=True,
                                        rounded=True,
                                        out_file=None)

        pydotplus.graph_from_dot_data(dot_data).write_png(filename)

    output_name = "outputs/classifiers/" + classifier_name + "_DT_graphviz.png"

    try:
        save_png_clf(DT, output_name, list(train_data.columns.values),
                     class_names1)
    except Exception as e:
        print(e)

    try:
        viz = dtreeviz(DT, train_data, y,
                       target_name=column_name,
                       feature_names=list(train_data.columns.values))
        viz.save(f"outputs/classifiers/{classifier_name}_decision_tree_viz.svg")

    except Exception as e:
        print(e)

    regr = DecisionTreeRegressor(criterion='mse', ccp_alpha=0.0006, random_state=42)
    regr.fit(train_data, y_regr)
    output_name = "outputs/classifiers/" + classifier_name + '_regressor_tree_mse_optimized.png'

    text_representation = tree.export_text(regr)
    print(text_representation)
    dot_data = tree.export_graphviz(regr, out_file=None,
                                    feature_names=list(train_data.columns.values),
                                    filled=True)
    graphviz.Source(dot_data, format="png")

    try:
        viz = dtreeviz(regr, train_data, y_regr,
                       target_name=column_name,
                       feature_names=list(train_data.columns.values),
                       class_names=[True, False])
        viz.save(f"outputs/classifiers/{classifier_name}_regressor_tree_mse_optimized.svg")
    except Exception as e:
        print(e)

    try:
        save_png_clf(regr, output_name, list(train_data.columns.values), [True, False])
    except Exception as e:
        print(e)


print("Try Decision Tree Classifier")
decisionTreeWork()


def RFWork():
    print("Try RF Classifier")

    RandomForestCl = RandomForestClassifier(bootstrap=False, max_depth=4, max_features='auto', min_samples_leaf=2,
                                            min_samples_split=2, n_estimators=17)
    output_name = "outputs/classifiers/" + classifier_name + "RF.log"

    RandomForestCl.fit(X_train, y_train.ravel())
    # print("oob_score = ", RandomForestCl.oob_score_)
    print(RandomForestCl.get_params())
    predictions3 = RandomForestCl.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    feature_names = list(train_data.columns.values)
    tree.plot_tree(RandomForestCl.estimators_[0],
                   feature_names=feature_names,
                   class_names=class_names1,

                   filled=True)
    fig.savefig(f"outputs/classifiers/{classifier_name}_rf_individualtree.png")

    p, r, f1, s = precision_recall_fscore_support(y_true=y_test, y_pred=predictions3, average='macro', warn_for=tuple())
    print("%.3f Precision\n%.3f Recall\n%.3f F1" % (p, r, f1))
    print(confusion_matrix(y_true=y_test, y_pred=predictions3))
    print(classification_report(y_true=y_test, y_pred=predictions3))


RFWork()

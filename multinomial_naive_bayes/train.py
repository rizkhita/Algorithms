# PENJELASAN FUNGSI LIBRARY
# pandas : penyedia fungsi operasi untuk mengelola data file menjadi data tabel (dataframe)
import pandas as pd
# read string sebagai file
from io import StringIO
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def baca_csv():
    dframe = pd.read_csv('taharah_intent.csv')
    return dframe

def convert_to_tidf():
    y = baca_csv()
    y['id_label'] = y['labels'].factorize()[0]
    id_label_df = y[['labels','id_label']].drop_duplicates().sort_values('id_label')
    label_ke_id = dict(id_label_df.values)
    id_ke_label = dict(id_label_df[['id_label', 'labels']].values)
    return y

# def mnb():
#     factory = StopWordRemoverFactory()
#     stop_word_list = factory.get_stop_words()
#     stop = stop_word_list + list(punctuation)
#     tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
#                             stop_words=stop)
#     df = convert_to_tidf()
#     X_train, X_test, y_train, y_test = train_test_split(df['questions'], df['labels'], random_state=0)
#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(X_train)
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     feed = MultinomialNB().fit(X_train_tfidf, y_train)
#     return feed, count_vect

#X_test.iloc[0]

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
factory = StopWordRemoverFactory()
stop_word_list = factory.get_stop_words()

# stopwords added
stopwords = stop_word_list + list(punctuation)
# create vectorizer
vect = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2', encoding='id', ngram_range=(1, 2), stop_words=stopwords)
# create data
df = convert_to_tidf()
features = vect.fit_transform(df.questions).toarray()
labels = df.id_label
# cross validation technique
X_train, X_test, y_train, y_test = train_test_split(df['questions'], df['labels'], random_state=0)

# import and instantiate CountVectorizer
# vect = CountVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='id', ngram_range=(1, 2),stop_words=stopwords)


# fit and transform X_train into X_train_dtm
X_train_dtm = vect.fit_transform(X_train)
# X_train_dtm.shape

"""
print("X_train")
for i in range(132):
    print(str([i])+" "+str(X_train.iloc[i]))

print("X_test")
for i in range(44):
    print(str([i])+" "+str(X_train.iloc[i]))
"""
# transform X_test into X_test_dtm
X_test_dtm = vect.transform(X_test)
# X_test_dtm.shape

# for prec
Y_train_dtm = vect.fit_transform(y_train)
# Y_train_dtm.shape

y_test_dtm = vect.transform(y_test)
# y_test_dtm.shape

# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics

print("\nAccuracy score :")
print(metrics.accuracy_score(y_test, y_pred_class))

y_test.value_counts()

# print the numbers of data
print("Trained : {0}".format(X_train.shape))
print("Tested: {0}".format(X_test.shape))

# print report
print("Report")
print(metrics.classification_report(y_test, y_pred_class))

# print the confusion matrix
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, y_pred_class))

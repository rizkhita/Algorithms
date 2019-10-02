# i used libraries that need to be installed first
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

def mnb():
    factory = StopWordRemoverFactory()
    stop_word_list = factory.get_stop_words()
    stop = stop_word_list + list(punctuation)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words=stop)
    df = convert_to_tidf()
    X_train, X_test, y_train, y_test = train_test_split(df['questions'], df['labels'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    feed = MultinomialNB().fit(X_train_tfidf, y_train)
    return feed, count_vect

#X_test.iloc[0]

def predict(question):
    feed, count_vect = mnb()
    intent = feed.predict(count_vect.transform([question]))
    intent = str(intent).strip("['']")
    return intent

question=input("Masukan pertanyaan : ")
x=predict(question)
intent=str(x).strip("['']")
print("Intent predicted : "+format(x))


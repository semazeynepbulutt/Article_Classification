from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import data_cleaning
import pandas as pd
import numpy as np
import pickle

def count_vectorizer(X_train: pd.core.series.Series ,X_test : pd.core.series.Series):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_test_counts = count_vect.transform(X_test)
    return X_train_counts,X_test_counts,count_vect

def tfidf(X_train_counts,X_test_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_train_tfidf,X_test_tfidf,tfidf_transformer

def prepare_data_for_training():
    df = data_cleaning.get_clean_data()
    articleSection_category = df.articleSection.astype('category')
    articleSection_target_dict = dict(enumerate(articleSection_category.cat.categories))
    df.articleSection = df.articleSection.astype('category').cat.codes
    df['target'] = df.articleSection.map(articleSection_target_dict)
    X = df['articleBody']
    y = df['articleSection']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) #different classes has different nuber of article
    X_train_counts, X_test_counts,count_vect = count_vectorizer(X_train=X_train,X_test=X_test)
    X_train_tfidf, X_test_tfidf,tfidf_transformer = tfidf(X_train_counts=X_train_counts,X_test_counts=X_test_counts)
    return articleSection_target_dict,X_train_tfidf, X_test_tfidf,y_train,y_test,count_vect,tfidf_transformer

def prediction_accuracy(model_name,X_test_tfidf,y_test):
    predicted = model_name.predict(X_test_tfidf)
    return np.mean(predicted == y_test)
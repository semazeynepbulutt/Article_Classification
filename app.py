from flask import Flask,jsonify,request
import pickle
from joblib import load
import pandas as pd


app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
app.secret_key = 'articleinfo'
articleSection_target_dict = {0: '3. Sayfa', 1: 'Dunya', 2: 'Ekonomi', 3: 'Futbol', 4: 'Kitap', 5: 'Magazin', 6: 'Otomobil', 7: 'Saglik', 8: 'Teknoloji'}

def create_prediction(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def create_option(option:str,article_body):
    print(option)
    print(article_body)
    series_data = pd.Series(article_body)
    if option == 'Multinominal Naive Bayes':
        clf_model = pickle.load(open('model_folder/model_mulinominalnb.sav', 'rb'))
        count_vectorizer = pickle.load(open('model_folder/count_vectorizer.pickle', 'rb'))
        tfidf_transformer = pickle.load(open('model_folder/tfidf_transformer.pickle', 'rb'))
        count_vectorizer_data = count_vectorizer.transform(series_data)
        tfidf_data = tfidf_transformer.transform(count_vectorizer_data)
        return jsonify({'prediction': articleSection_target_dict.get(clf_model.predict(tfidf_data)[0])})

    elif option == 'SVM':
        clf_model = pickle.load(open('model_folder/sgd_classifier.sav', 'rb'))
        count_vectorizer = pickle.load(open('model_folder/count_vectorizer_svm.pickle', 'rb'))
        tfidf_transformer = pickle.load(open('model_folder/tfidf_transformer_svm.pickle', 'rb'))
        count_vectorizer_data = count_vectorizer.transform(series_data)
        tfidf_data = tfidf_transformer.transform(count_vectorizer_data)
        return jsonify({'prediction': articleSection_target_dict.get(clf_model.predict(tfidf_data)[0])})
    else:
        clf_model = load('model_folder/xgboost_classifier.joblib')
        count_vectorizer = load('model_folder/count_vectorizer_xgb.joblib')
        tfidf_transformer = load('model_folder/tfidf_transformer_xgb.joblib')
        count_vectorizer_data = count_vectorizer.transform(series_data)
        tfidf_data = tfidf_transformer.transform(count_vectorizer_data)
        return jsonify({'prediction': articleSection_target_dict.get(clf_model.predict(tfidf_data)[0])})

@app.route('/article_info/<string:option>/article_body',methods = ['POST'])
def create_prediction(option):
    request_data = request.get_json()
    article_body = request_data['article_body']
    return create_option(option=option,article_body=article_body)

app.run(port=5005,debug=True)


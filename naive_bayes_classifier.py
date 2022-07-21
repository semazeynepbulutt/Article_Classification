from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import data_for_training
import pickle
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

articleSection_target_dict,X_train_tfidf, X_test_tfidf,y_train,y_test,count_vect,tfidf_transformer = data_for_training.prepare_data_for_training()
def fit_model():
    global X_train_tfidf
    global y_train
    model_multinominalnb = MultinomialNB().fit(X_train_tfidf, y_train)
    return model_multinominalnb

def create_prediction(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def create_classification_report(model_name:str,y_test,y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    name = model_name + '.xlsx'
    df.to_excel(name)

def heatmap(model_name,y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    fig = ax.get_figure()
    name = model_name + '_heatmap_cm.png'
    fig.savefig(name)

if __name__ == '__main__':
    model_mulinominalnb = fit_model()
    y_pred = create_prediction(model=model_mulinominalnb,X_test=X_test_tfidf)
    create_classification_report(model_name='naive_bayes',y_test=y_test,y_pred=y_pred)
    filename = 'model_mulinominalnb.sav'
    pickle.dump(model_mulinominalnb, open(filename, 'wb'))
    filename = 'count_vectorizer.pickle'
    pickle.dump(count_vect, open(filename, 'wb'))
    filename = 'tfidf_transformer.pickle'
    pickle.dump(tfidf_transformer, open(filename, 'wb'))
    heatmap(model_name='multinominal_naive_bayes', y_test=y_test, y_pred=y_pred)
    print(articleSection_target_dict)

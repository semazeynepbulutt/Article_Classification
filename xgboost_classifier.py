import seaborn as sns
import data_for_training
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBClassifier
from joblib import dump


articleSection_target_dict,X_train_tfidf, X_test_tfidf,y_train,y_test,count_vect,tfidf_transformer = data_for_training.prepare_data_for_training()

def model():
    classifier = XGBClassifier(tree_method='gpu_hist')
    model_xgb = classifier.fit(X_train_tfidf, y_train)
    return model_xgb

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
    model_xgb = model()
    y_pred = create_prediction(model=model_xgb,X_test=X_test_tfidf)
    create_classification_report(model_name='xgboost_classifier',y_test=y_test,y_pred=y_pred)
    dump(model_xgb, 'xgboost_classifier.joblib')
    dump(count_vect, 'count_vectorizer_xgb.joblib')
    dump(tfidf_transformer, 'tfidf_transformer_xgb.joblib')
    heatmap(model_name='rf_classifier',y_test=y_test,y_pred=y_pred)
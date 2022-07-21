import seaborn as sns
import data_for_training
import pickle
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


articleSection_target_dict,X_train_tfidf, X_test_tfidf,y_train,y_test,count_vect,tfidf_transformer = data_for_training.prepare_data_for_training()
def model():
    model_sgdclassifier = SGDClassifier()
    return model_sgdclassifier
def grid_search_cv(model):
    hyperparameter_space = {'loss': ['log'],
                            'penalty': ['l2', 'l1', 'elasticnet'],
                            'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.001, 0.002, 0.003, 0.004, 0.005,
                                      0.01],
                            'fit_intercept': [True, False],
                            'max_iter': list(range(100, 2000, 100))}

    gs = GridSearchCV(model, param_grid=hyperparameter_space,
                      scoring="accuracy",
                      n_jobs=-1, cv=10, return_train_score=True)

    gs.fit(X_train_tfidf, y_train)
    return gs.best_estimator_,gs.best_params_

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
    model_sgd = model()
    best_sgd_model,best_param = grid_search_cv(model=model_sgd)
    y_pred = create_prediction(model=best_sgd_model,X_test=X_test_tfidf)
    create_classification_report(model_name='sgd_classifier',y_test=y_test,y_pred=y_pred)
    filename = 'sgd_classifier.sav'
    pickle.dump(best_sgd_model, open(filename, 'wb'))
    filename = 'count_vectorizer_svm.pickle'
    pickle.dump(count_vect, open(filename, 'wb'))
    filename = 'tfidf_transformer_svm.pickle'
    pickle.dump(tfidf_transformer, open(filename, 'wb'))
    heatmap(model_name='sgd_classifier',y_test=y_test,y_pred=y_pred)
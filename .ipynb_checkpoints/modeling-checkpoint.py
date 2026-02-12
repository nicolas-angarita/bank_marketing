import pandas as pd
import numpy as np

import wrangle as w

import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score



def model_split(train, validate, test):

    X_train= train.drop(columns = ['job','marital','education','default', 'housing', 'loan'
                                   ,'contact','month','duration','poutcome','y','y_encoded'])
    y_train = train.y_encoded
    
    
    X_validate = validate.drop(columns = ['job','marital','education','default', 'housing',
                                          'loan','contact','month','duration','poutcome','y','y_encoded'])
    y_validate = validate.y_encoded
    
    X_test = test.drop(columns = ['job','marital','education','default', 'housing',
                                  'loan','contact','month','duration','poutcome','y','y_encoded'])
    y_test = test.y_encoded



    return X_train, y_train, X_validate, y_validate, X_test, y_test


def decision_tree(X_train, y_train, X_validate, y_validate):
    
    metrics = []
    
    for i in range(2,11):
        
        clf = DecisionTreeClassifier(max_depth = i, random_state = 42)
        
        clf.fit(X_train, y_train)
    
        # Predictions
        
        y_train_preds = clf.predict(X_train)
        y_validate_preds = clf.predict(X_validate)       
        y_validate_proba = clf.predict_proba(X_validate)[:,1]
    
        # Accuracy
        train_accuracy = accuracy_score(y_train, y_train_preds)
        validate_accuracy = accuracy_score(y_validate, y_validate_preds)
    
        # precision, recall, f1, auc
        precision = precision_score(y_validate, y_validate_preds)
        recall = recall_score(y_validate, y_validate_preds)
        f1 = f1_score(y_validate, y_validate_preds)
        auc = roc_auc_score(y_validate, y_validate_proba)
    
        output = {
            'max_depth': i,
            'train_accuracy': train_accuracy,
            'validate_accuracy': validate_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }
    
        metrics.append(output)
    
    df_dt = pd.DataFrame(metrics)
    df_dt['difference'] = df_dt.train_accuracy - df_dt.validate_accuracy
    
    df_dt = df_dt.sort_values('precision', ascending=False).reset_index(drop= 'index')    

    return df_dt


def random_forest(X_train, y_train, X_validate, y_validate):

    metrics = []
    max_depth = 11
    
    for i in range(1, max_depth):
        depth = max_depth - i
        n_samples = i
        rf = RandomForestClassifier(max_depth = depth, min_samples_leaf = n_samples,
                                    random_state = 42, n_estimators = 200)
    
        rf.fit(X_train, y_train)
    
    
        # Predictions
        y_train_pred = rf.predict(X_train)
        y_validate_pred = rf.predict(X_validate)
        y_validate_prob = rf.predict_proba(X_validate)[:, 1]
    
        # Accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    
        # precision, recall, f1, auc
        precision = precision_score(y_validate, y_validate_pred)
        recall = recall_score(y_validate, y_validate_pred)
        f1 = f1_score(y_validate, y_validate_pred)
        auc = roc_auc_score(y_validate, y_validate_prob)
    
        output = {
            'max_depth': depth,
            'min_samples_leaf': n_samples,
            'train_accuracy': train_accuracy,
            'validate_accuracy': validate_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }
    
        metrics.append(output)
    
    df_rf = pd.DataFrame(metrics)
    df_rf['difference'] = df_rf.train_accuracy - df_rf.validate_accuracy
    
    df_rf = df_rf.sort_values('precision', ascending=False).reset_index(drop= 'index')

    return df_rf


def knn(X_train, y_train, X_validate, y_validate):
    
    metrics = []
    
    for i in range(1,21):
        
        knn = KNeighborsClassifier(n_neighbors=i)
        
        knn.fit(X_train,y_train)
        
        y_train_preds = knn.predict(X_train)
        y_validate_preds = knn.predict(X_validate)
        y_validate_proba = knn.predict_proba(X_validate)[:,1]
        
        train_accuracy = accuracy_score(y_train,y_train_preds)
        validate_accuracy = accuracy_score(y_validate,y_validate_preds)
        
        precision = precision_score(y_validate, y_validate_preds)
        recall = recall_score(y_validate, y_validate_preds)
        f1 = f1_score(y_validate, y_validate_preds)
        roc_auc = roc_auc_score(y_validate, y_validate_proba)
    
        output = { 
               'neighbors': i, 
               'train_accuracy': train_accuracy,
               'validate_accuracy': validate_accuracy,
               'precision': precision,
               'recall': recall,
               'f1_score': f1,
               'roc_auc': roc_auc
                }
        
        metrics.append(output)
    
    df_knn = pd.DataFrame(metrics)
    df_knn['difference'] = df_knn.train_accuracy - df_knn.validate_accuracy 
    df_knn = df_knn.sort_values('precision', ascending= False).reset_index(drop= 'index')

    return df_knn


def logistic_regresion(X_train, y_train, X_validate, y_validate):
    
    metrics = []
    c_reg = [0.01, 0.1, 1, 10]
    
    for i in c_reg:
    
        log = LogisticRegression(max_iter = 2000, class_weight = 'balanced',C = i, random_state = 1722)
    
        log.fit(X_train, y_train)
    
        
        y_train_preds = log.predict(X_train)
        y_validate_preds = log.predict(X_validate)
        y_validate_proba = log.predict_proba(X_validate)[:,1]
        
        train_accuracy = accuracy_score(y_train,y_train_preds)
        validate_accuracy = accuracy_score(y_validate,y_validate_preds)
        
        precision = precision_score(y_validate, y_validate_preds)
        recall = recall_score(y_validate, y_validate_preds)
        f1 = f1_score(y_validate, y_validate_preds)
        roc_auc = roc_auc_score(y_validate, y_validate_proba)
    
        
        metrics.append({
            "regularization": i,
            "train_accuracy": train_accuracy,
            "validate_accuracy": validate_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        })
    
        
    
    df_log = pd.DataFrame(metrics)
    df_log['difference'] = df_log.train_accuracy - df_log.validate_accuracy 
    df_log = df_log.sort_values('precision', ascending= False).reset_index(drop= 'index')

    return df_log

def best_model(X_train, y_train, X_validate, y_validate):

    df_dt = decision_tree(X_train, y_train, X_validate, y_validate)
    df_rf = random_forest(X_train, y_train, X_validate, y_validate)
    df_knn = knn(X_train, y_train, X_validate, y_validate)
    df_log = logistic_regresion(X_train, y_train, X_validate, y_validate)

    
    models = {'decision_tree':df_dt,'random_forest':df_rf,'knn':df_knn,'logistic_regression':df_log}
    best_model = []
    
    
    for name, model in models.items():
        top_model = model.loc[0,'train_accuracy':'difference']
    
        row_dict = top_model.to_dict()
        
        row_dict['model'] = name
        
        best_model.append(row_dict)
    
    best_performing_models = pd.DataFrame(best_model)
    model_col = best_performing_models.pop('model')
    best_performing_models.insert(0, 'model', model_col)
    
    best_performing_models = best_performing_models.sort_values('precision', ascending = False).reset_index(drop = 'index')
       
    return best_performing_models


def model_scores(X_train, y_train, X_val, y_val):
    '''
    Score multiple models on train and validate datasets.
    Print classification reports to look at the best model to test on.
    Returns each different trained model.
    models = decision_tree_model, random_forest_model, knn_model, logistic_regression_model
    '''
    
    decision_tree_model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
    
    random_forest_model = RandomForestClassifier(max_depth = 9, min_samples_leaf = 2, n_estimators=200, random_state = 42)
    
    knn_model = KNeighborsClassifier(n_neighbors = 6)
    
    logistic_regression_model = LogisticRegression(max_iter= 2000, C=.10,class_weight='balanced', random_state = 1722)
    
    models = [decision_tree_model, random_forest_model, knn_model, logistic_regression_model]
    
    # within this for loop I fitted each model, 
    # got my actual values and my predicted values for each model
    # printed my train and validate score and classification report

    for model in models:
        model.fit(X_train, y_train)
        actual_train = y_train
        predicted_train = model.predict(X_train)
        actual_val = y_val
        predicted_val = model.predict(X_val)
        
    '''print(model)
        print('\n')
        print('Train Score: ')
        print(classification_report(actual_train, predicted_train))
        print('Validate Score: ')
        print(classification_report(actual_val, predicted_val))
        print('--------------------------------------------------------------')
        print('\n')
        '''
    return decision_tree_model, random_forest_model, knn_model, logistic_regression_model


def test_model(X_train, y_train, X_validate, y_validate, X_test, y_test):

    decision_tree_model, random_forest_model, knn_model, logistic_regression_model = model_scores(X_train, y_train, X_validate, y_validate)
    
    best_performing_models = best_model(X_train, y_train, X_validate, y_validate)
    
    test_y_pred = random_forest_model.predict(X_test)
    
    test_precision = precision_score(y_test,test_y_pred)
    
    best_performing_models.loc[0, 'test_precision'] = test_precision
    
    return best_performing_models, best_performing_models.loc[0]
import sys

from scipy.sparse.construct import random

sys.path.append('/Users/tristenmarto/Documents/Studie/Thesis/Synthsonic_data_analysis')
result_path = '/Users/tristenmarto/Documents/Studie/Thesis/Synthsonic_data_analysis/CSV_results/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smote_variants as sv
import logging

from collections import Counter

import csv

import xgboost as xgb

from imblearn.datasets import fetch_datasets

from Evaluation import synthsonic

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# metrics 
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (recall_score, roc_auc_score, confusion_matrix, precision_score, precision_recall_curve,
                             f1_score, balanced_accuracy_score, accuracy_score, average_precision_score)


# functions
def load_data(name) :
    
    data = datasets[name]

    X,y,title = data['data'], data['target'], data['DESCR']
    
    return X, y, title

def metrics_test(dataset_name, X_train, X_test, y_train, y_test, clf) :
        
    clf.fit(X_train,y_train)
    
    prediction = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]

    f_dict = {
        'name':dataset_name,
        'balanced acc': balanced_accuracy_score(y_test,prediction),
        'precision': precision_score(y_test,prediction),
        'recall': recall_score(y_test,prediction),
        'f1_score': f1_score(y_test, prediction, average='binary'),
        'geometric mean': geometric_mean_score(y_test, prediction),
        'average precision' : average_precision_score(y_test, prediction),
        'clf':clf,
        'prediction': prediction,
        'proba': proba
    }
    
    return f_dict

def metrics_test_samples(dataset_name, samples, X_test, y_test, clf) :
    
    res = []
    
    for gen in samples :
        
        prop,x,y = gen['prop'],gen['x_sample'],gen['y_sample']
    
        clf.fit(x,y)

        prediction = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:,1]

        f_dict = {
            'name':dataset_name,
            'proportion':prop,
            'balanced acc': balanced_accuracy_score(y_test,prediction),
            'precision': precision_score(y_test,prediction),
            'recall': recall_score(y_test,prediction),
            'f1_score': f1_score(y_test, prediction, average='binary'),
            'geometric mean': geometric_mean_score(y_test, prediction),
            'average precision' : average_precision_score(y_test, prediction),
            'clf':clf,
            'prediction': prediction,
            'proba': proba
        }
        
        res.append(f_dict)
        
    return res

def test_dataset(name) :
    
    X,y,title = load_data(name)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = random_state)
        
    result = metrics_test(title, X_train, X_test, y_train, y_test, clf)
        
    return result

def test_dataset_samples(name, proportions) :
    
    X,y,title = load_data(name)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = random_state)
    
    # create samples
    samples = sampler(oversampler, X_train,y_train, proportions)
    
    result = metrics_test_samples(title, samples, X_test, y_test, clf)
    
    return result

def run_multiple(list_datasets,filename) :
    
    results = []
    
    for dataset in list_datasets :
        
        print(dataset)
        
        results.append(test_dataset(dataset))
    
    df = pd.DataFrame(results)
    
    df.to_csv(result_path+filename+".csv")
    
    return df


def run_multiple_samples(oversampler, proportions, list_datasets, filename) :
    
    for dataset in list_datasets :
        
        print(dataset)    
        tempres = test_dataset_samples(dataset, proportions)

        with open(result_path+filename+".csv", "a") as f :

            writer = csv.writer(f)
            writer.writerow(tempres)




def sampler(oversampler, x, y, proportions) :
    
    samples = []
    
    for prop in proportions :
        o = oversampler(proportion = prop,
                        distinct_threshold = 20,
                        random_state = random_state)
    
        x_g, y_g = o.sample(x,y)
    
        f_dict = {
            'prop':prop,
            'x_sample':x_g,
            'y_sample':y_g
        }

        samples.append(f_dict)
    
    return samples


if __name__ == "__main__" :

    random_state = 5

    datasets = fetch_datasets()

    newlist = ['thyroid_sick', 'coil_2000', 'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2', 'webpage', 'ozone_level', 'mammography', 'protein_homo', 'abalone_19']

    oversampler = synthsonic
    clf = xgb.XGBClassifier(verbosity=0)
    proportions = [0.1,0.3,0.5,0.7,0.9,1]


    res = run_multiple_samples(oversampler = oversampler, 
                               proportions = proportions, 
                               list_datasets = newlist, 
                               filename = synthsonic.__name__)

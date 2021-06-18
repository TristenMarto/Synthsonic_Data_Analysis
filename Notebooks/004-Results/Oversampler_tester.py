import sys

from scipy.sparse.construct import random

sys.path.append('/Users/tristenmarto/Documents/Studie/Thesis/Synthsonic_data_analysis')
result_path = '/Users/tristenmarto/Documents/Studie/Thesis/Synthsonic_data_analysis/CSV_results/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smote_variants as sv
import logging
import time

from collections import Counter

import csv

import xgboost as xgb

from imblearn.datasets import fetch_datasets

from Evaluation import synthsonic

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler

# metrics 
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (recall_score, roc_auc_score, confusion_matrix, precision_score, precision_recall_curve,
                             f1_score, balanced_accuracy_score, accuracy_score, average_precision_score)



# functions
def load_data(name) :
    
    datasets = fetch_datasets()

    data = datasets[name]

    X,y,title = data['data'], data['target'], data['DESCR']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, title

def tester(outputfile, datasets,oversamplers=None,proportions=[0], random_state = 5, include_base = True, clf = xgb.XGBClassifier(use_label_encoder=False, verbosity=0)) :

    res_df = pd.DataFrame()

    failed_sets = []

    for dataset in datasets :
        X,y,title = load_data(dataset)
        print(15*"-")
        print(title)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = random_state)


        if include_base :
            temp_res = metrics_test(dataset, X_train,X_test,y_train,y_test)
            res_df = res_df.append(temp_res)
            
        if oversamplers is not None :
            for oversampler in oversamplers :
                for prop in proportions :
                    print(f"oversampling with {oversampler.__name__} with proportion {prop}")
                    #sampletje genereren
                    
                    if oversampler.__name__ == 'synthsonic' :
                        o = oversampler(random_state=random_state,
                                        distinct_threshold = 20,
                                        proportion = prop)

                    else :

                        try :
                            o = oversampler(random_state=random_state,
                                            sampling_strategy = prop)

                        except :
                            print("cant instantiate with these params, setting default")
                            o = oversampler(random_state=random_state)


                    try :
                        x_g, y_g = o.sample(X_train,y_train)
                    
                        temp_res = metrics_test(dataset = dataset, 
                                                X_train = x_g, 
                                                X_test = X_test, 
                                                y_train = y_g, 
                                                y_test = y_test, 
                                                clf = clf,
                                                oversampler = oversampler.__name__, 
                                                prop = prop)

                        res_df = res_df.append(temp_res)

                    except AttributeError :
                        x_g, y_g = o.fit_resample(X_train,y_train)

                        temp_res = metrics_test(dataset = dataset, 
                                                X_train = x_g, 
                                                X_test = X_test, 
                                                y_train = y_g, 
                                                y_test = y_test, 
                                                clf = clf,
                                                oversampler = oversampler.__name__, 
                                                prop = prop)
                        res_df = res_df.append(temp_res)
                    
                    except ValueError:
                        print(f"Could not run {oversampler.__name__} for {dataset}, n_components must be between 0 and min(n_samples, n_features)")
                        failed_sets.append(dataset)

                    res_df.to_csv(result_path+outputfile+".csv")

    if len(failed_sets) is not 0 :
        print("Did not run following sets: ", failed_sets)

    return res_df

def metrics_test(dataset, X_train,X_test,y_train,y_test, clf = xgb.XGBClassifier(use_label_encoder=False, verbosity=0), oversampler=None, prop=0, random_state=5) :


    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]

    f_dict = {
        'name':dataset,
        'oversampler':oversampler,
        'proportion': prop,
        'balanced acc': balanced_accuracy_score(y_test,prediction),
        'precision': precision_score(y_test,prediction),
        'recall': recall_score(y_test,prediction),
        'f1_score': f1_score(y_test, prediction, average='binary'),
        'geometric mean': geometric_mean_score(y_test, prediction),
        'average precision' : average_precision_score(y_test, prediction)
    }
        
    df_res = pd.DataFrame([f_dict])

    return df_res

if __name__== "__main__" :

    datasets = fetch_datasets()
    sets = list(datasets.keys())
    oversamplers = [RandomOverSampler]#sv.SMOTE, sv.Borderline_SMOTE1, sv.ADASYN, sv.polynom_fit_SMOTE, sv.SVM_balance, sv.Random_SMOTE]
    props = [0.3, 0.5, 0.7, 0.9, 1]
    outputfile = "randomOversampling"

    df = tester(outputfile, sets, oversamplers, props)

    print(df)

from networkx.generators.classic import balanced_tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from imblearn.datasets import fetch_datasets
from tqdm import tqdm
from sv_synthsonic import synthsonic
from handle_data import HandleData
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# oversamplers
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE
import smote_variants as sv

# classifiers
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#metrics
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import (recall_score, roc_auc_score, confusion_matrix, precision_score, precision_recall_curve,
                             f1_score, balanced_accuracy_score, accuracy_score, auc)

def configure_oversampler(dataset_info, oversampler, proportion) :

    oversampler.categorical_features = dataset_info['cat_columns']
    oversampler.sampling_strategy = oversampler.proportion = proportion
    if dataset_info['minority'] < dataset_info['features']:
        oversampler.do_PCA = False
        oversampler.ordering = ''
    
    return oversampler

def fit_predict(clf, X_train, X_test, y_train, y_test) :

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:,1]
    pr, rec, _ = precision_recall_curve(y_test, probas)
    
    f_dict = {
        'balanced_accuracy': balanced_accuracy_score(y_test, prediction),
        'G_mean': geometric_mean_score(y_test, prediction),
        'f1': f1_score(y_test, prediction, average='binary'),
        'precision': precision_score(y_test, prediction),
        'recall': recall_score(y_test, prediction),
        'pr_auc': auc(rec,pr)
    }

    return f_dict

def cross_validate_oversampler(dataset, oversampler, proportion, n_splits, clf) :

    df = pd.DataFrame({
        'dataset': dataset, 
        'oversampler': oversampler,
        'proportion': proportion},
         index=[0])
    
    # loading of dataset
    X,y,_ = HandleData().load_data(dataset)
    # stratified cross validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5)
    
    balanced_accuracy = []
    G_mean = []
    f1 = []
    precision = []
    recall = []
    pr_auc = []
    runtimes = []

    for fold, (train_index, test_index) in enumerate(kf.split(X,y), 1):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try :
            start_time = time.time()
            X_g, y_g = oversampler.sample(X_train, y_train)
            runtime = time.time() - start_time
        
        except AttributeError :
            start_time = time.time()
            X_g, y_g = oversampler.fit_resample(X_train, y_train)
            runtime = time.time() - start_time

        # fit, predict and score
        temp_res = fit_predict(clf, X_g, X_test, y_g, y_test)
        balanced_accuracy.append(temp_res['balanced_accuracy'])
        G_mean.append(temp_res['G_mean'])
        f1.append(temp_res['f1'])
        precision.append(temp_res['precision'])
        recall.append(temp_res['recall'])
        pr_auc.append(temp_res['pr_auc'])
        runtimes.append(runtime)

    df['balanced_accuracy'] = np.mean(balanced_accuracy)
    df['balanced_accuracy_std'] = np.std(balanced_accuracy)
    df['G_mean'] = np.mean(G_mean)
    df['G_mean_std'] = np.std(G_mean)
    df['f1'] = np.mean(f1)
    df['f1_std'] = np.std(f1)
    df['precision'] = np.mean(precision)
    df['precision_std'] = np.std(precision)
    df['recall'] = np.mean(recall)
    df['recall_std'] = np.std(recall)
    df['pr_auc'] = np.mean(pr_auc)
    df['pr_auc_std'] = np.std(pr_auc)
    df['runtime'] = np.mean(runtimes)
    df['runtime_std'] = np.std(runtimes) 

    return df

def perform_tests(respath, datasets, oversamplers, proportions, n_splits, clf) :

    temp_dfs = []

    for dataset in tqdm(datasets, desc='dataset') :
        dataset_info = HandleData().analyse_dataset(dataset, proportions)
        for oversampler in oversamplers :
            for proportion in dataset_info['possible_proportions'] :
                oversampler = configure_oversampler(dataset_info, oversampler, proportion)
                temp_dfs.append(cross_validate_oversampler(dataset, oversampler, proportion, n_splits, clf))

    df = pd.concat(temp_dfs)
    df.to_csv(respath.joinpath('cross_validation2.csv'), index=False)

    return df

def main() :

    random_state = 5
    n_splits = 5
    clf = XGBClassifier(verbosity=0, use_label_encoder=False)
    datasets = list(fetch_datasets().keys())
    oversamplers = [RandomOverSampler(random_state=random_state), 
                    SMOTE(random_state=random_state, n_jobs=-1),
                    SMOTENC(random_state=random_state, categorical_features = [], n_jobs=-1), 
                    SVMSMOTE(random_state=random_state, n_jobs=-1), 
                    ADASYN(random_state=random_state, n_jobs=-1), 
                    BorderlineSMOTE(random_state=random_state, n_jobs=-1),
                    synthsonic(distinct_threshold=20),
                    sv.polynom_fit_SMOTE(random_state=random_state),
                    sv.Random_SMOTE(random_state=random_state)]
    proportions = np.array([0.2, 0.4, 0.6, 0.8, 1])

    fail = []
    temp_dfs = []
    currentpath=Path.cwd()
    cachepath = currentpath.joinpath('CSV_results', 'Cross_validation', 'Cache')
    respath = currentpath.joinpath('CSV_results', 'Cross_validation')

    perform_tests(respath, datasets, oversamplers, proportions, n_splits, clf)

    print("Done")

if __name__ == "__main__" :
    main()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from imblearn.datasets import fetch_datasets
from tqdm import tqdm
from sv_synthsonic import synthsonic, load_data, pca_plot
from pathlib import Path
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
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

def analyse_dataset(dataset, n_splits) :
    
    X,y,title = load_data(dataset)

    # categorical check
    cat_columns = []
    num_columns = []

    for i in range(X.shape[1]) :
        if len(np.unique(X[:,i])) < 20 :
            cat_columns.append(i)

        else :
            num_columns.append(i)

    v,c = np.unique(y,return_counts = True)
    majority = c[0]
    minority = c[1]
    min_prop = minority/majority

    f_dict = {
        'title': title,
        'samples': X.shape[0],
        'features': X.shape[1],
        'cat_columns': cat_columns,
        'cat_features': len(cat_columns),
        'num_columns': num_columns,
        'num_features': len(num_columns),
        'minority': minority,
        'min_proportion': min_prop
    }

    return f_dict

def configure_oversampler(dataset_info, oversampler, proportion) :

    oversampler.categorical_features = dataset_info['cat_columns']

    # Turn off PCA when samples < features
    if dataset_info['minority'] < dataset_info['features']:
        oversampler.do_PCA = False 

    oversampler.sampling_strategy = oversampler.proportion = proportion
    
    return oversampler

def cross_validate_oversampler(dataset, oversampler, proportion, n_splits, clf) :

    df = pd.DataFrame({
        'dataset':dataset, 
        'oversampler': oversampler,
        'proportion' : proportion},
         index=[0])
    
    # loading of dataset
    X,y,_ = load_data(dataset)
    
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
        clf.fit(X_g,y_g)
        prediction = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:,1]

        pr, rec, _ = precision_recall_curve(y_test, probas)

        balanced_accuracy.append(balanced_accuracy_score(y_test, prediction))
        G_mean.append(geometric_mean_score(y_test, prediction))
        f1.append(f1_score(y_test, prediction, average='binary'))
        precision.append(precision_score(y_test, prediction))
        recall.append(recall_score(y_test, prediction))
        pr_auc.append(auc(rec,pr))
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

if __name__ == "__main__" :

    random_state = 5
    n_splits = 5
    clf = XGBClassifier(verbosity=0, use_label_encoder=False)
    datasets = list(fetch_datasets().keys())
    oversamplers = [RandomOverSampler(random_state=random_state), 
                    SMOTE(random_state=random_state, n_jobs=-1),
                    # SMOTENC(random_state=random_state, n_jobs=-1), 
                    SVMSMOTE(random_state=random_state, n_jobs=-1), 
                    ADASYN(random_state=random_state, n_jobs=-1), 
                    BorderlineSMOTE(random_state=random_state, n_jobs=-1),
                    synthsonic(random_state=random_state, distinct_threshold=20),
                    sv.polynom_fit_SMOTE(random_state=random_state),
                    sv.Random_SMOTE(random_state=random_state)]
    proportions = np.array([0.2, 0.4, 0.6, 0.8, 1])

fail = []
temp_dfs = []

for dataset in tqdm(datasets[4:], desc="dataset") :
    
    print(f"analysing dataset {dataset}.")
    datasetdfs=[]
    # prepare and analyse data
    dataset_info = analyse_dataset(dataset, n_splits)

    possible_proportions = proportions[proportions > dataset_info['min_proportion']]

    for oversampler in tqdm(oversamplers, desc="oversampler") :
        
        print(f"Generating samples for {oversampler}.")
        
        for proportion in possible_proportions :
            
            try :
                oversampler = configure_oversampler(dataset_info, oversampler, proportion)

            except :
                fail.append((dataset, oversampler, proportion))
                continue
            
            temp_dfs.append(cross_validate_oversampler(dataset, oversampler, proportion, n_splits, clf))

    df2 = pd.concat(temp_dfs)
    df2.to_csv("/Users/tristenmarto/Documents/Studie/Thesis/Synthsonic_data_analysis/CSV_results/Cross_validation/Cache/"+dataset+"_cross_val_xgboost.csv", index=False)

df = pd.concat(temp_dfs)
df.to_csv("/Users/tristenmarto/Documents/Studie/Thesis/Synthsonic_data_analysis/CSV_results/Cross_validation/cross_val_xgboost.csv", index=False)

print("Done")
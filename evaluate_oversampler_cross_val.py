import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from tqdm import tqdm
from handle_data import HandleData
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier

#metrics
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (recall_score, precision_score, precision_recall_curve,
                             f1_score, balanced_accuracy_score, auc)

class cross_validation :

    def __init__(self,
                 kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=5),
                 clf = XGBClassifier(verbosity=0, use_label_encoder=False)) :

        self.kf = kf
        self.clf = clf
        
    def configure_oversampler(self, dataset_info, oversampler, proportion) :

        oversampler.categorical_features = dataset_info['cat_columns']
        oversampler.numerical_features = dataset_info['num_columns']
        oversampler.sampling_strategy = oversampler.proportion = proportion
        if dataset_info['minority'] < dataset_info['features']:
            oversampler.do_PCA = False
            oversampler.ordering = ''
        
        return oversampler

    def fit_predict(self, X_train, X_test, y_train, y_test) :

        self.clf.fit(X_train, y_train)
        prediction = self.clf.predict(X_test)
        probas = self.clf.predict_proba(X_test)[:,1]
        pr, rec, _ = precision_recall_curve(y_test, probas)
        
        return {
            'balanced_accuracy': balanced_accuracy_score(y_test, prediction),
            'G_mean': geometric_mean_score(y_test, prediction),
            'f1': f1_score(y_test, prediction, average='binary'),
            'precision': precision_score(y_test, prediction),
            'recall': recall_score(y_test, prediction),
            'pr_auc': auc(rec,pr)
        }

    def generate_from_oversampler(self, oversampler, X, y) :

        try :
            X_g, y_g = oversampler.fit_resample(X, y) 

        except AttributeError:
            logging.exception('')
            X_g, y_g = oversampler.sample(X, y)

        return X_g, y_g 

    def calculate_pr_curve(self, X, y, dataset_name, oversampler, proportion) :
        
        dfinfo = HandleData().analyse_dataset(X,y,dataset_name, np.array([0.2,0.4,0.6,0.8,1.]))
        oversampler = self.configure_oversampler(dfinfo, oversampler, proportion)

        y_real = []
        y_proba = []

        for fold, (train_index, test_index) in enumerate(self.kf.split(X,y), 1) :

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_g, y_g = self.generate_from_oversampler(oversampler, X_train, y_train)

            self.clf.fit(X_g, y_g)
            probas = self.clf.predict_proba(X_test)
            precision, recall, _ = precision_recall_curve(y_test, probas[:,1])
            y_real.append(y_test)
            y_proba.append(probas[:,1])
        
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)

        return precision_recall_curve(y_real, y_proba) 

    def cross_validate_oversampler(self, X, y, dataset_name, oversampler, proportion) :

        df = pd.DataFrame({
            'dataset': dataset_name, 
            'oversampler': oversampler,
            'proportion': proportion},
            index=[0])
                
        balanced_accuracy = []
        G_mean = []
        f1 = []
        precision = []
        recall = []
        pr_auc = []
        runtimes = []

        for fold, (train_index, test_index) in enumerate(self.kf.split(X,y), 1):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start_time = time.time()
            X_g, y_g = self.generate_from_oversampler(oversampler, X_train, y_train)
            runtime = time.time() - start_time

            # fit, predict and score
            temp_res = self.fit_predict(X_g, X_test, y_g, y_test)
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

    def loop_over_datasets(self, datasets, proportions, oversampler) :
        
        dfs = []

        for dataset in datasets :
            X,y,title = dataset['data'].values, dataset['target'].values, dataset['DESCR']
            y = HandleData().label_encode(y)
            dataset_info = HandleData().analyse_dataset(X, y, title, proportions)

            for proportion in dataset_info['possible_proportions'] :
                oversampler = self.configure_oversampler(dataset_info, oversampler, proportion)
                dfs.append(self.cross_validate_oversampler(X, y, title, oversampler, proportion))

        return pd.concat(dfs)

    def loop_over_proportions(self, X, y, dataset_name, oversampler, proportions) :

        dataset_info = HandleData().analyse_dataset(X, y, dataset_name, proportions)
        dfs = []

        for proportion in dataset_info['possible_proportions'] :
            oversampler = self.configure_oversampler(dataset_info, oversampler, proportion)
            dfs.append(self.cross_validate_oversampler(X, y, dataset_name, oversampler, proportion))

        return pd.concat(dfs)

    def loop_over_oversamplers(self, X, y, dataset_name, oversamplers, proportions) :

        dfs = []

        for oversampler in tqdm(oversamplers, desc='oversampler') :
            dfs.append(self.loop_over_proportions(X, y, dataset_name, oversampler, proportions))

        return pd.concat(dfs)
            
    def full_test(self, datasets, oversamplers, proportions) :

        dfs = []

        for dataset in tqdm(datasets, desc='dataset') :
            X,y,dataset_name = dataset['data'], dataset['target'], dataset['DESCR']
            y = HandleData().label_encode(y) 
            dfs.append(self.loop_over_oversamplers(X, y, dataset_name, oversamplers, proportions))

        return pd.concat(dfs)
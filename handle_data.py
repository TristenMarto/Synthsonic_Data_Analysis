import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.datasets import fetch_datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class HandleData :
    
    def load_data(self, name) :
        
        datasets = fetch_datasets()
        data = datasets[name]
        X,y,title = data['data'], data['target'], data['DESCR']
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        return X, y_enc, title

    def pca_plot(self, X, y, title="") :
        
        x = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        principal_components = PCA(n_components=2).fit_transform(x)
        labels,count = np.unique(y, return_counts=True)

        f, ax = plt.subplots(figsize = (12,7))
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f'2 component PCA {title}')

        for l in labels :
            rowix = np.where(y==l)[0]
            ax.scatter(principal_components[rowix,0]
                    , principal_components[rowix,1]
                    , label=f"Class {l}: {count[l]}"
                    , color=f'C{l}')

        ax.legend()
        plt.show()

    def divide_num_cat(self, X, threshold) :

        cat_columns = []
        num_columns = []

        for i in range(X.shape[1]) :
            if len(np.unique(X[:,i])) < threshold :
                cat_columns.append(i)
            else :
                num_columns.append(i)

        return cat_columns, num_columns

    def find_min_maj(self, y) :

        v,c = np.unique(y, return_counts = True)
        majority = c[0]
        minority = c[1]
        min_prop = minority/majority

        return majority, minority, min_prop

    def label_encode(self, y) :

        return LabelEncoder().fit_transform(y)

    def label_encode_df(self, df, columns) :

        for label in columns :
            df[label] = LabelEncoder().fit_transform(df[label])

        return df

    def analyse_dataset(self, X, y, dataset_name, proportions) :
        
        cat_columns, num_columns = self.divide_num_cat(X, 20)
        majority, minority, min_prop = self.find_min_maj(y)
        possible_props = proportions[proportions > min_prop] 

        return {
            'title': dataset_name,
            'samples': X.shape[0],
            'features': X.shape[1],
            'cat_columns': cat_columns,
            'cat_features': len(cat_columns),
            'num_columns': num_columns,
            'num_features': len(num_columns),
            'minority': minority,
            'possible_proportions': possible_props
        }
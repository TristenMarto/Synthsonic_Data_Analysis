import numpy as np
import matplotlib.pyplot as plt
import smote_variants as sv
import imbalanced_databases as imbd
import pandas as pd
import time
import logging

from collections import Counter

from imblearn.datasets import fetch_datasets
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced

from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

_logger = logging.getLogger('smote_variants')
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)

class synthsonic(sv.OverSampling) :
# Implementation of the synthsonic package in Smote Variants
    
    def __init__(self,
                 proportion=1.0,
                 distinct_threshold=-1,
                 do_PCA = True, 
                 random_state=None) :
        
        
        super().__init__()
        
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.distinct_threshold = distinct_threshold
        self.n_jobs = n_jobs
        self.do_PCA = do_PCA

        self.random_state = random_state
        
    @classmethod
    def parameter_combinations(cls, raw=False) :
        
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        
        return cls.generate_parameter_combinations(parameter_combinations, raw)
    
    def sample(self, X, y) :
        
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())
        
        # Find minority class
        # self.find_minority(X,y)
        self.class_label_statistics(X, y)
        
        self.X_min = X[y == self.min_label]
        
        
        # fit model
        kde = KDECopulaNNPdf(distinct_threshold=self.distinct_threshold,
                             do_PCA=self.do_PCA,
                             numerical_columns=[],
                             categorical_columns=[])
        kde.fit(self.X_min)
        
        # determine n_samples
        self.n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        
        # sample
        x1 = kde.sample_no_weights(n_samples=self.n_to_sample, mode='cheap')
        
        X_samp = np.vstack([X,x1])
        y_samp = np.hstack([y, [self.min_label]*self.n_to_sample])
        
        return X_samp, y_samp

    def set_params(self, **params):
        
        for param, value in params.items() :
            setattr(self, param, value)

    def get_params(self) :
        return {'proportion': self.proportion, 
                'distinct_threshold':self.distinct_threshold,
                'random_state': self.random_state,
                'do_PCA': self.do_PCA}
    


def writetodict(dataset,name) :
    
    data = dataset.iloc[:,:-1].values
    target = dataset.iloc[:,-1].values

    return {'data':data,
            'target':target,
            'name':name}
    
def load_data(name) :
    
    datasets = fetch_datasets()

    data = datasets[name]

    X,y,title = data['data'], data['target'], data['DESCR']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, title


def pca_plot(X,y) :
    
    x = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    
    fig = plt.figure(figsize = (12,7))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(f'2 component PCA')

    counter = Counter(y)

    for label, _ in counter.items() :
        rowix = np.where(y == label)[0]
        ax.scatter(principalComponents[rowix, 0], principalComponents[rowix, 1], label=str(label))

    ax.legend()

    fig.show()
    print(counter)

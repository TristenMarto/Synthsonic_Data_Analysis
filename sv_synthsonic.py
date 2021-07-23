import numpy as np
import pandas as pd
import smote_variants as sv
import logging
from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf

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
                 ordering = 'pca', 
                 X_min = None,
                 numerical_features = [],
                 categorical_features = [],
                 clf = None) :
        
        super().__init__()
        
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.proportion = proportion
        self.X_min = []
        
        # Synthsonic parameters
        self.distinct_threshold = distinct_threshold
        self.do_PCA = do_PCA
        self.ordering = ordering
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.clf = clf
    
    @classmethod
    def parameter_combinations(cls, raw=False) :
        
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        
        return cls.generate_parameter_combinations(parameter_combinations, raw)
    
    def sample(self, X, y) :
        
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())
        
        # determine minority and majority class
        self.class_label_statistics(X, y)
        self.X_min = X[y == self.min_label]
        self.n_to_sample = self.det_n_to_sample(self.proportion,
                                                self.class_stats[self.maj_label],
                                                self.class_stats[self.min_label])

        kde = KDECopulaNNPdf(distinct_threshold = self.distinct_threshold,
                            do_PCA = self.do_PCA,
                            ordering = self.ordering,
                            numerical_columns = self.numerical_features,
                            categorical_columns = self.categorical_features,
                            clf = self.clf)
        kde.fit(self.X_min)
        
        #sample
        x1 = kde.sample_no_weights(n_samples=self.n_to_sample, mode='cheap')
        X_samp = np.vstack([X,x1])
        y_samp = np.hstack([y, [self.min_label]*self.n_to_sample])
        
        return X_samp, y_samp

    def set_params(self, **params):
        
        for param, value in params.items() :
            setattr(self, param, value)

    def get_params(self) :
        
        return {'proportion': self.proportion,
                'distinct_threshold': self.distinct_threshold,
                'do_PCA': self.do_PCA,
                'ordering': self.ordering}
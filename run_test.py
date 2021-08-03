from evaluate_oversampler_cross_val import cross_validation
import numpy as np
import pandas as pd
from pathlib import Path
from handle_data import HandleData
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# oversamplers
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE
from sv_synthsonic import synthsonic
import smote_variants as sv

# classifiers
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def main() :

    datafolder = Path.cwd() / 'Datasets'
    df = pd.read_csv(datafolder.joinpath('credit_fraud.csv'))
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    title = 'Credit_Fraud'
    random_state=5
    oversamplers = [RandomOverSampler(random_state=random_state), 
                    SMOTE(random_state=random_state, n_jobs=-1),
                    SMOTENC(categorical_features=[],random_state=random_state),
                    SVMSMOTE(random_state=random_state, n_jobs=-1), 
                    ADASYN(random_state=random_state, n_jobs=-1), 
                    BorderlineSMOTE(random_state=random_state, n_jobs=-1),
                    synthsonic(distinct_threshold=20),
                    sv.polynom_fit_SMOTE(random_state=random_state),
                    sv.Random_SMOTE(random_state=random_state),
                    sv.NoSMOTE()]
    proportions = np.array([0.2, 0.4, 0.6, 0.8, 1.])
    cv = cross_validation()
    res = cv.loop_over_oversamplers(X,y,title, oversamplers, proportions)
    respath = Path.cwd() / 'CSV_results' / 'Cross_validation' / 'cross_validation_credit_fraud.csv'
    res.to_csv(respath)

    print("Done")

if __name__ == "__main__" :
    main()
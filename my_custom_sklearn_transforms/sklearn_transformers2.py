from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import pandas as pd

class BalanceData(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_perfil0 = X[X["OBJETIVO"] == 'Aceptado']
        df_perfil1 = X[X["OBJETIVO"] == 'Sospechoso']

        samples = 8000

        df_perfil0_upsampled = resample(df_perfil0,
                                        replace=True,  # sample with replacement
                                        n_samples=samples,  # to match majority class
                                        random_state=123)  # reproducible results

        df_perfil1_upsampled = resample(df_perfil1,
                                        replace=True,  # sample with replacement
                                        n_samples=samples,  # to match majority class
                                        random_state=123)  # reproducible results

        return pd.concat([df_perfil0_upsampled, df_perfil1_upsampled])

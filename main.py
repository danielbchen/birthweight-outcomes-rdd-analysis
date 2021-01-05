import numpy as np
import os
import pandas as pd
import plotnine #pip install plotnine 
import statsmodels.formula.api as smf


path = os.path.dirname(os.path.abspath("__file__"))
def main():
    """
    """

    df = birth_weight_loader()
    print('\n\n')
    print('The table below summarizes the distribution of new born birthweights:',
          '\n\n',
          get_descriptive_stats(df, 'bweight'))


def birth_weight_loader():
    """Loads in stata file containg birth weight data."""

    data_path = os.path.join(path, 'almond_etal_2008.dta')
    df = pd.read_stata(data_path)

    return df 


def get_descriptive_stats(dataframe, column):
    """Returns descriptive statistics for a specified column in dataframe."""

    df = dataframe.copy()
    descriptive_stats = df[column].describe()

    return descriptive_stats

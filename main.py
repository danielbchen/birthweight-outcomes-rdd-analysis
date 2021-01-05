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
    df = column_cutter(df)


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


def column_cutter(dataframe):
    """XX"""

    df = dataframe.copy()

    bins = [
        1329.9,
        1358.25,
        1386.6,
        1414.95,
        1443.3,
        1471.65,
        1500,
        1528.35,
        1556.7,
        1585.05,
        1613.4,
        1641.75,
        1670.1,
    ]
    labels = []
    df['bweight_bins'] = pd.cut(df['bweight'], bins=bins, right=False)

    return df

    grouped = df.groupby('bweight_bins').agg({'agedth5': 'mean'}).reset_index()
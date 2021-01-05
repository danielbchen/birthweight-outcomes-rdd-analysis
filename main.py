import numpy as np
import os
import pandas as pd
from plotnine import ggplot, geom_point, aes, theme, element_text, labs  # pip install plotnine
import statsmodels.formula.api as smf


path = os.path.dirname(os.path.abspath("__file__"))
def main():
    """
    """

    # Loads in data
    df = birth_weight_loader() 
    print('\n\n')
    # Returns descriptive stats
    print('The table below summarizes the distribution of new born birthweights:',
          '\n\n',
          get_descriptive_stats(df, 'bweight'))
    # Cuts birth weight column into bins
    df = column_cutter(df)
    # Groups/summarizes mortality data for plotting
    mortality_aggregations = {'agedth5': 'mean', 'agedth4': 'mean'}
    mortality_summary = data_grouper(df, 'bweight_bins', mortality_aggregations)
    # Plot mortality rates by birth weight
    plotter(mortality_summary, 'bweight_bins', 'agedth5',
            'Mean One-Year Mortality Rate by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Mortality Rate')


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
    """Groups birth weight column into bins."""

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
    df['bweight_bins'] = pd.cut(df['bweight'], bins=bins, right=False)

    return df


def data_grouper(dataframe, grouper_col, agg_dict):
    """Returns a new dataframe with original data grouped by specified 
    arguments.
    """

    df = dataframe.copy()
    grouped_df = df.groupby(grouper_col).agg(agg_dict).reset_index()

    return grouped_df


def plotter(dataframe, x, y, title, x_axis_label, y_axis_label):
    """Returns a scatter plot using specified data parameters."""

    df = dataframe.copy()
    plot = (ggplot(df, aes(x, y)) +
            geom_point() +
            theme(axis_text_x=element_text(rotation=50, hjust=1)) +
            labs(title=title,
                 x=x_axis_label,
                 y=y_axis_label)
            )
    
    return plot
import numpy as np
import os
import pandas as pd
# pip install plotnine
import plotnine as p9
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
    plotter(mortality_summary, 'bweight_bins', 'agedth4',
            'Mean 28-Day Mortality Rate by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Mortality Rate')
    # Plot background covariates against birth weight
    background_covariates = ['mom_age', 'mom_ed1', 'gest', 'nprenatal', 'yob']
    summary_stat = [['mean'] * 5]
    summary_stat = [mean for sublist in summary_stat for mean in sublist]
    background_aggregations = dict(zip(background_covariates, summary_stat))
    background_summary = data_grouper(
        df, 'bweight_bins', background_aggregations)

    plotter(background_summary, 'bweight_bins', 'mom_age',
            'Mean Age of Mother by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Age of Mother')
    plotter(background_summary, 'bweight_bins', 'mom_ed1',
            'Mother Edu Less than HS by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Mother Edu')
    plotter(background_summary, 'bweight_bins', 'gest',
            'Mean Gestational Age by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Mean Gestational Age')
    plotter(background_summary, 'bweight_bins', 'nprenatal',
            'Mean Number of Prenatal Care Visits by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Mean Number of Prenatal Visits Age')
    plotter(background_summary, 'bweight_bins', 'yob',
            'Mean Birth Year by Birth Weight',
            'Birth Weight Bin (Grams)',
            'Mean Year of Birth')
    # Run OLS on background characteristics
    df = regression_column_creator(df)
    background_discontinuity = run_rdd(dataframe=df,
                                       dep_vars=['mom_age', 'mom_ed1',
                                                 'gest', 'nprenatal', 'yob'],
                                       ind_vars=['alpha_1',
                                                 'alpha_2', 'alpha_3'],
                                       caliper=85)
    # Run OLS discontinuity on mortality outcomes
    mortality_discontinuity = run_rdd(dataframe=df,
                                      dep_vars=['agedth5', 'agedth4'],
                                      ind_vars=['alpha_1',
                                                'alpha_2', 'alpha_3'],
                                      caliper=85)


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
    plot = (p9.ggplot(df, p9.aes(x, y)) +
            p9.geom_point() +
            p9.theme(axis_text_x=p9.element_text(rotation=50, hjust=1)) +
            p9.labs(title=title,
                    x=x_axis_label,
                    y=y_axis_label) +
            p9.geom_vline(xintercept=6.5, size=2)
    )
    
    return plot
    #p9.ggsave(plot=plot, filename='{}.png'.format(title), dpi=1000)


def regression_column_creator(dataframe):
    """Derives new columns that will be used as regressors for OLS."""

    df = dataframe.copy()

    df['alpha_1'] = [0 if weight >= 1500 else 1 for weight in df['bweight']]
    df['threshold_distance'] = df['bweight'] - 1500
    df['alpha_2'] = df['alpha_1'] * df['threshold_distance']
    df['alpha_3'] = (1 - df['alpha_1']) * (df['threshold_distance'])

    return df


def parameter_getter(list_object, parameter, n_regressors):
    """Returns a list of specified parameters from a regression output. 
    
    Keyword arguments:
    1) list_object must be a list of Regression Results Wrappers
    2) parameter must be a string. 'coefficient' returns the estimated
       betas for a regressor. 'standard error' returns the corrresponding 
       standard errors. 'p-value' returns the corresponding p-values. 
    3) n-regressors must be an integer representing the number of variables 
       on the right hand side of the regression. 
    """

    if parameter == 'coefficient':
        raw_values = [item.params for item in list_object]
    elif parameter == 'standard error':
        raw_values = [item.bse for item in list_object]
    elif parameter == 'p-value':
        raw_values = [item.pvalues for item in list_object]

    values = []
    for i in range(len(list_object)):
        for j in range(n_regressors + 1):
            values.append(raw_values[i][j])

    indexes_to_drop = [number for number in range(len(values) + 1)]
    indexes_to_drop = indexes_to_drop[0:len(indexes_to_drop) - 4:4]

    parameters = np.delete(values, indexes_to_drop)

    return parameters
    

def run_rdd(dataframe, dep_vars, ind_vars, caliper):
    """Runs and regression discontinuity via OLS given data, edogenous 
    variable(s), exogenous variable(s), and a caliper. 
    """

    df = dataframe.copy()

    df = df[(df['threshold_distance'] >= (caliper * -1)) &
            (df['threshold_distance'] <= caliper)]

    variables = dep_vars
    right_hand_side = ' + '.join([var for var in ind_vars])
    formulas = [var + ' ~ ' + right_hand_side for var in variables]

    regressions = [smf.ols(formula=formula, data=df).fit()for formula in formulas]
    
    dep_vars_col = [[var] * len(ind_vars) for var in dep_vars]
    dep_vars_col = [var for sublist in dep_vars_col for var in sublist]
    ind_vars_col = ind_vars * len(dep_vars)

    results = pd.DataFrame({
        'OUTCOME_VAR': dep_vars_col,
        'IND_VAR': ind_vars_col,
        'ESTIMATE': parameter_getter(regressions, 'coefficient', len(ind_vars)),
        'STD_ERROR': parameter_getter(regressions, 'standard error', len(ind_vars)),
        'P_VALUE': parameter_getter(regressions, 'p-value', len(ind_vars))
    })

    return results

"""
The issue with the function above is that there are binary variables that each
produce a coefficient. For example, when we include the variable 'mom_race', we 
don't only get one variable returned. Instead, we get a coefficient for both
black and other, probably relative to the third option which gets automatically 
dropped as a comparison group. 
"""
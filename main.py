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


def birth_weight_loader():
    """Loads in stata file containg birth weight data."""

    data_path = os.path.join(path, 'almond_etal_2008.dta')
    df = pd.read_stata(data_path)

    return df 
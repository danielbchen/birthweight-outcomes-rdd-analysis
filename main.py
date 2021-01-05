import numpy as np
import os
import pandas as pd
import plotnine #pip install plotnine 
import statsmodels.formula.api as smf


path = os.path.dirname(os.path.abspath("__file__"))
data_path = os.path.join(path, 'almond_etal_2008.dta')
df = pd.read_stata(data_path)
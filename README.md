# Project Overview

Health care economists question whether the benefits of added medical care outweigh the costs. [Almond et al. (2008)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2903901/) propose a Regression Discontinuity Design (RDD) to compare health outcomes of newborns born around the threshold of 1500 grams. Those born under 1500 grams are considered very low birth weight and are consequently receive supplemental medical attention. Those born above the threshold are generally considered healthy. Notably, 1500 grams is a conventional measure not necessarily rooted in biology. 

This project provides an overview of RDD, evaluates its assumptions, re-produces findings from Almond et al., and presents thoughts on increased medical expenditures.  

# Regression Discontinuity Design Primer

## Background

RDD is common in political science and econometrics as a quasi-experimental design when randomization is not possible. The idea is to exploit a threshold in which units above and below the margin are assigned to "treatment" and "control" groups. Units that lie very close to the threshold are similar enough, on average, which allows researchers to identify the missing counterfactual required to estimate average treatment effects. As a result they may not vary much in terms of potential outcomes, and any differences in outcomes may be attributed to an intervention.

## Assumption

The key assumption for RDD to be valid is that the mean potential outcomes are continuous on the running variable. In other words, there must not be any discontinuous jump in our outcome variable when it is plotted against the variable that determines assignment to treatment or control plotted on the x-axis. This assumption will be contextualized and evaluated later on.

# Loading and Checking Data

While [matplotlib](https://matplotlib.org) and [seaborn](https://seaborn.pydata.org) are popular plotting libraries for Python, I turn to [plotnine](https://plotnine.readthedocs.io/en/stable/#) for this project. My original analysis was in R, and ggplot created clear visualizations which I admired, and I wished to create them in Python. Plotnine is Python's version of R's ggplot, and the library utilizes a practically identical grammar of graphics. The library will need to be installed before running.

The original reserachers use data from the National Center for Health Statistics (NCHS) which contains data on 66 million births between 1983 a


```python
import numpy as np
import os
import pandas as pd
# pip install plotnine
from plotnine import *
import statsmodels.formula.api as smf

path = os.path.dirname(os.path.abspath("__file__"))
```


```python
def birth_weight_loader():
    """Loads in stata file containg birth weight data."""

    data_path = os.path.join(path, 'almond_etal_2008.dta')
    df = pd.read_stata(data_path)

    return df 
```


```python
df = birth_weight_loader() 
df
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yob</th>
      <th>yod</th>
      <th>staters</th>
      <th>mom_age</th>
      <th>mom_race</th>
      <th>mom_ed</th>
      <th>mom_ed1</th>
      <th>mom_ed2</th>
      <th>mom_ed3</th>
      <th>mom_ed4</th>
      <th>...</th>
      <th>dad_age</th>
      <th>dad_race</th>
      <th>sex</th>
      <th>plural</th>
      <th>mom_origin</th>
      <th>dad_origin</th>
      <th>tot_order</th>
      <th>live_order</th>
      <th>pldel</th>
      <th>attend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1983</td>
      <td>NaN</td>
      <td>1</td>
      <td>21</td>
      <td>black</td>
      <td>12.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20.0</td>
      <td>black</td>
      <td>2</td>
      <td>1</td>
      <td>88</td>
      <td>88</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1983</td>
      <td>NaN</td>
      <td>1</td>
      <td>34</td>
      <td>white</td>
      <td>12.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>37.0</td>
      <td>white</td>
      <td>2</td>
      <td>1</td>
      <td>88</td>
      <td>88</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1983</td>
      <td>NaN</td>
      <td>10</td>
      <td>31</td>
      <td>white</td>
      <td>12.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>25.0</td>
      <td>white</td>
      <td>2</td>
      <td>1</td>
      <td>88</td>
      <td>88</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>NaN</td>
      <td>1</td>
      <td>18</td>
      <td>black</td>
      <td>11.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>other</td>
      <td>2</td>
      <td>1</td>
      <td>88</td>
      <td>88</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1983</td>
      <td>NaN</td>
      <td>1</td>
      <td>17</td>
      <td>black</td>
      <td>9.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>other</td>
      <td>2</td>
      <td>1</td>
      <td>88</td>
      <td>88</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>376403</th>
      <td>2002</td>
      <td>2002.0</td>
      <td>50</td>
      <td>33</td>
      <td>white</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>32.0</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>376404</th>
      <td>2002</td>
      <td>2002.0</td>
      <td>50</td>
      <td>19</td>
      <td>black</td>
      <td>10.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>other</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>376405</th>
      <td>2002</td>
      <td>2003.0</td>
      <td>50</td>
      <td>34</td>
      <td>white</td>
      <td>12.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>white</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>376406</th>
      <td>2002</td>
      <td>2003.0</td>
      <td>50</td>
      <td>26</td>
      <td>white</td>
      <td>13.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>white</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
    <tr>
      <th>376407</th>
      <td>2002</td>
      <td>2002.0</td>
      <td>51</td>
      <td>44</td>
      <td>white</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>other</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>Hospital Births</td>
      <td>Physician</td>
    </tr>
  </tbody>
</table>
<p>376408 rows × 44 columns</p>
</div>



Let's start by getting an idea of the distribution of birth weights in the sample.


```python
def get_descriptive_stats(dataframe, column):
    """Returns descriptive statistics for a specified column in dataframe."""

    df = dataframe.copy()
    descriptive_stats = df[column].describe()

    return descriptive_stats
```


```python
get_descriptive_stats(dataframe=df, column='yob')
```




    count    376408.000000
    mean       1992.986289
    std           6.181239
    min        1983.000000
    25%        1987.000000
    50%        1995.000000
    75%        1999.000000
    max        2002.000000
    Name: yob, dtype: float64



# Visualizing Mortality Rates by Birth Weights

Findings from RDD depend on a discontinuity around a threshold in which the difference can potentially be attributed to an intervention. Below I group newborns into birth weight groups that are equally spaced above and below 1500 grams.


```python
def column_cutter(dataframe):
    """Groups birth weight column into bins."""

    df = dataframe.copy()

    bins = [
        1329.9, 1358.25, 1386.6, 1414.95,
        1443.3, 1471.65, 1500,   1528.35,
        1556.7, 1585.05, 1613.4, 1641.75,
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
            theme_bw() +
            theme(axis_text_x=element_text(rotation=50, hjust=1)) +
            labs(title=title,
                    x=x_axis_label,
                    y=y_axis_label) +
            geom_vline(xintercept=6.5, size=2) 
    )
    
    return plot
```


```python
df = column_cutter(dataframe=df)
mortality_aggregations = {'agedth5': 'mean', 
                          'agedth4': 'mean'}
mortality_summary_df = data_grouper(dataframe=df, 
                                    grouper_col='bweight_bins', 
                                    agg_dict=mortality_aggregations)
mortality_summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bweight_bins</th>
      <th>agedth5</th>
      <th>agedth4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1329.9, 1358.25)</td>
      <td>0.065423</td>
      <td>0.047998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[1358.25, 1386.6)</td>
      <td>0.077292</td>
      <td>0.056158</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[1386.6, 1414.95)</td>
      <td>0.069375</td>
      <td>0.049445</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[1414.95, 1443.3)</td>
      <td>0.063181</td>
      <td>0.044574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[1443.3, 1471.65)</td>
      <td>0.061195</td>
      <td>0.042713</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[1471.65, 1500.0)</td>
      <td>0.056921</td>
      <td>0.039149</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[1500.0, 1528.35)</td>
      <td>0.061541</td>
      <td>0.043902</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[1528.35, 1556.7)</td>
      <td>0.054012</td>
      <td>0.037291</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[1556.7, 1585.05)</td>
      <td>0.050504</td>
      <td>0.033744</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[1585.05, 1613.4)</td>
      <td>0.052595</td>
      <td>0.036606</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[1613.4, 1641.75)</td>
      <td>0.045852</td>
      <td>0.030820</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[1641.75, 1670.1)</td>
      <td>0.044650</td>
      <td>0.028939</td>
    </tr>
  </tbody>
</table>
</div>




```python
yearly_plot = plotter(dataframe=mortality_summary_df, x='bweight_bins', y='agedth5',
                      title='Mean One-Year Mortality Rate by Birth Weight',
                      x_axis_label='Birth Weight Bin (Grams)',
                      y_axis_label='Mortality Rate')

monthly_plot = plotter(dataframe=mortality_summary_df, x='bweight_bins', y='agedth4',
                       title='Mean 28-Day Mortality Rate by Birth Weight',
                       x_axis_label='Birth Weight Bin (Grams)',
                       y_axis_label='Mortality Rate')

yearly_plot, monthly_plot
```


![png](output_17_0.png)



![png](output_17_1.png)





    (<ggplot: (8780569230561)>, <ggplot: (8780569833133)>)



In looking at the plots above, there is a negative relationship between birth weight and the mortality rate. As a newborn's birth weight increases, the mortality rate falls. Intuitively, this makes sense given that newborns who weigh more likely face fewer health complications. 

However, upon visual inspection of both the yearly and monthly mortality rate plots, there appears to be a *discontinuity* as we move from the bin directly below (1471.65 grams inclusive to 1500 grams exclusive) the 1500 gram threshold to the bin directly above the threshold (1500 grams inclusive to 1528.35 grams exclusive). In moving from left to the right on the x-axis, the mortality rate generally declines until reaching the threshold where it jumps up before falling back down again. 

In looking back at the grouped dataframe, the one-year mortality rate jumps from about 5.69% to 6.15%. When looking at the 28-day mortality rate, the rate jumps from about 3.91% to 4.39%. 

With this in mind, it's possible that the difference in the mortality rate in the vicinity of the threshold may be attributed to alternative treatments that newborns above and below the cutoff receive. 

# Evaluating the Assumption

As a reminder, for RDD estimates to hold, our assignment variable must not be manipulable. If we were able to manipulate the running variable, then any discontinuous jumps in the vicinity of the threshold may be attributed to manipulation and not treatment effects. 

Our running variable is birth weight. For our assumption to hold, birth weight cannot be manipulated. This is *likely met*. As the original researchers argue, birth weight cannot be manipulated to the degree of precision necessary to invalidate this assumption. According to [Pressman et al. (2000)](https://www.sciencedirect.com/science/article/abs/pii/S0029784499006171), birthweight cannot be predicted before birth with the necessary precision to change the newborn’s classification of above 1500 grams or below 1500 grams. In other words, while parents and doctors may have some influence in a baby’s birthweight either through maternal diet or inducing early birth, it is not plausible that they have the precision or control required to change the birthweight categorization of the newborn.

# Plotting the Background Covariates

Next, I'll plot mother's education, mother's age, mother's education less than high school, gestational age, prenatal care visits, and year of birth against birth weight. The idea here is to also expect smoothness around the threshold so that discontinuities around the threshold are again attributed treatment and not to a background characteristic of the mother (or some other variable).


```python
background_covariates = ['mom_age', 'mom_ed1', 'gest', 'nprenatal', 'yob']
summary_stat = [['mean'] * 5]
summary_stat = [mean for sublist in summary_stat for mean in sublist]
background_aggregations = dict(zip(background_covariates, summary_stat))
background_summary = data_grouper(dataframe=df, 
                                  grouper_col='bweight_bins', 
                                  agg_dict=background_aggregations)
background_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bweight_bins</th>
      <th>mom_age</th>
      <th>mom_ed1</th>
      <th>gest</th>
      <th>nprenatal</th>
      <th>yob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1329.9, 1358.25)</td>
      <td>26.764292</td>
      <td>0.240599</td>
      <td>31.111612</td>
      <td>8.692638</td>
      <td>1993.778661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[1358.25, 1386.6)</td>
      <td>26.211801</td>
      <td>0.260021</td>
      <td>31.225998</td>
      <td>8.367048</td>
      <td>1992.710015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[1386.6, 1414.95)</td>
      <td>26.437522</td>
      <td>0.247293</td>
      <td>31.401115</td>
      <td>8.635597</td>
      <td>1992.938866</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[1414.95, 1443.3)</td>
      <td>26.460619</td>
      <td>0.250876</td>
      <td>31.548761</td>
      <td>8.746612</td>
      <td>1993.088415</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[1443.3, 1471.65)</td>
      <td>26.487695</td>
      <td>0.249645</td>
      <td>31.745317</td>
      <td>8.818672</td>
      <td>1993.026664</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[1471.65, 1500.0)</td>
      <td>26.436943</td>
      <td>0.250489</td>
      <td>31.899631</td>
      <td>8.908753</td>
      <td>1993.083331</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[1500.0, 1528.35)</td>
      <td>26.387245</td>
      <td>0.253558</td>
      <td>32.136983</td>
      <td>8.917313</td>
      <td>1992.913425</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[1528.35, 1556.7)</td>
      <td>26.452241</td>
      <td>0.252207</td>
      <td>32.348546</td>
      <td>9.074383</td>
      <td>1993.141750</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[1556.7, 1585.05)</td>
      <td>26.454008</td>
      <td>0.245460</td>
      <td>32.453237</td>
      <td>9.112997</td>
      <td>1993.041816</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[1585.05, 1613.4)</td>
      <td>26.424861</td>
      <td>0.248136</td>
      <td>32.626248</td>
      <td>9.074674</td>
      <td>1992.944753</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[1613.4, 1641.75)</td>
      <td>26.488966</td>
      <td>0.248733</td>
      <td>32.753542</td>
      <td>9.195656</td>
      <td>1992.991954</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[1641.75, 1670.1)</td>
      <td>26.378687</td>
      <td>0.246321</td>
      <td>32.988830</td>
      <td>9.302794</td>
      <td>1992.865211</td>
    </tr>
  </tbody>
</table>
</div>




```python
mom_age_plot = plotter(dataframe=background_summary, x='bweight_bins', y='mom_age',
                       title='Mean Age of Mother by Birth Weight',
                       x_axis_label='Birth Weight Bin (Grams)',
                       y_axis_label='Age of Mother')
    
mom_hs_ed_plot = plotter(dataframe=background_summary, x='bweight_bins', y='mom_ed1',
                         title='Mother Edu Less than HS by Birth Weight',
                         x_axis_label='Birth Weight Bin (Grams)',
                         y_axis_label='Mother Edu')
    
gestation_plot = plotter(dataframe=background_summary, x='bweight_bins', y='gest',
                         title='Mean Gestational Age by Birth Weight',
                         x_axis_label='Birth Weight Bin (Grams)',
                         y_axis_label='Mean Gestational Age')
    
prenatal_plot = plotter(dataframe=background_summary, x='bweight_bins', y='nprenatal',
                         title='Mean Number of Prenatal Care Visits by Birth Weight',
                         x_axis_label='Birth Weight Bin (Grams)',
                         y_axis_label='Mean Number of Prenatal Visits Age')
    
yob_plot = plotter(dataframe=background_summary, x='bweight_bins', y='yob',
                   title='Mean Birth Year by Birth Weight',
                   x_axis_label='Birth Weight Bin (Grams)',
                   y_axis_label='Mean Year of Birth')

mom_age_plot, mom_hs_ed_plot, gestation_plot, prenatal_plot, yob_plot
```


![png](output_24_0.png)



![png](output_24_1.png)



![png](output_24_2.png)



![png](output_24_3.png)



![png](output_24_4.png)





    (<ggplot: (8780569227925)>,
     <ggplot: (8780568989673)>,
     <ggplot: (8780568996825)>,
     <ggplot: (8780568870105)>,
     <ggplot: (8780568871145)>)



From the plots above, gestational age and number of prenatal care visits show smooth, clear trends around the cutoff. However, mother's age, mother's education less than high school, and birth year are more erratic, suggesting discontinuities around the cutoff. With this in mind, we can control for these variables by adding them as regressors later on in the analysis. In the next section, I run the regression without these covariates. 

# A Formal Test for Smoothness

After glancing at the plots, it appears that there is discontinuity around the 1500 gram threshold. We can empirically test if this is actually the case by estimating the following regression: 

$B_i = a_0 + a_1VLBW + a_2VLBW_i(g_i - 1500) + a_3(1 - VLBW_i)(g_i - 1500) + e_i$

where:
- $B_i$ is the background covariate
- $VLBW_i$ is a binary indicator for whether or not a newborn is classified as very low birth weight (strictly less than 1500 grams)
- $g_i$ is the birth weight
- $e_i$ is the error term

and the coefficient:
- $a_1$ measures the gap of the discontinuity, if any
- $a_2$ is the slope of the line for newborns categorized as very low birth weight. When VLBW is "activated" or equal to 1, our equation simplifies to $a_0 + a_1 + a_2(g_i - 1500)$ with $a_0$ and $a_1$ being constants and $g_i - 1500$ being a transformation of our running variable, birth weight (the independent variable), we are basically left with a line in the form of $Y = mx + b$.
- $a_3$ is the slope of the line for newborns not categorized as very low birth weight. When VLBW is not "activated' or equal to 0 our equation simplifies to $a_0 + a_3(g_i - 1500)$. Again, this is simply a linear line.

Note how the difference in the simplified equations in bullet three and four are nearly identical. The assuming that $a_2$ and $a_3$ are identical, the only difference between the two reduced equations is $a_1$ which consequently must be the jump from the line with $a_2$ as the slope to the line with $a_3$ as the slope.

The following functions below will derive columns for the regressors and return a dataframe summarizing the regression output from statsmodels.


```python
def regression_column_creator(dataframe):
    """Derives new columns that will be used as regressors for OLS."""

    df = dataframe.copy()

    df['alpha_1'] = [0 if weight >= 1500 else 1 for weight in df['bweight']]
    df['threshold_distance'] = df['bweight'] - 1500
    df['alpha_2'] = df['alpha_1'] * df['threshold_distance']
    df['alpha_3'] = (1 - df['alpha_1']) * (df['threshold_distance'])

    return df


def regression_to_dataframe(estimates, standard_errors, p_values):
    """Transforms statsmodels ols output into dataframe."""
    
    df = pd.DataFrame({
        'ESTIMATE': estimates,
        'STD_ERROR': standard_errors,
        'P_VALUE': p_values
    })

    return df


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

    regressions = [smf.ols(formula=formula, data=df).fit() for formula in formulas]

    dataframes = []
    for regression in regressions:
        dataframes.append(regression_to_dataframe(
            regression.params, regression.bse, regression.pvalues))

    results = (pd.concat(dataframes)
                 .reset_index()
                 .rename(columns={'index': 'EXOGENOUS_VARIABLE'})
    )

    iterations = len(results['EXOGENOUS_VARIABLE'].unique())
    exog_vars = [[var] * iterations for var in dep_vars]
    exog_col = [var for sublist in exog_vars for var in sublist]
    results['ENDOGENOUS_VARIABLE'] = exog_col

    results = results[[
        'ENDOGENOUS_VARIABLE',
        'EXOGENOUS_VARIABLE',
        'ESTIMATE',
        'STD_ERROR',
        'P_VALUE'
    ]]

    coefs = ['alpha_1', 'alpha_2', 'alpha_3']
    results = results[results['EXOGENOUS_VARIABLE'].isin(coefs)]
    
    return results
```


```python
df = regression_column_creator(df)
run_rdd(dataframe=df,
        dep_vars=['mom_age', 'mom_ed1','gest', 'nprenatal', 'yob'],
        ind_vars=['alpha_1','alpha_2', 'alpha_3'],
        caliper=85)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENDOGENOUS_VARIABLE</th>
      <th>EXOGENOUS_VARIABLE</th>
      <th>ESTIMATE</th>
      <th>STD_ERROR</th>
      <th>P_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>mom_age</td>
      <td>alpha_1</td>
      <td>0.241361</td>
      <td>0.060065</td>
      <td>5.862731e-05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mom_age</td>
      <td>alpha_2</td>
      <td>0.001799</td>
      <td>0.000889</td>
      <td>4.312242e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mom_age</td>
      <td>alpha_3</td>
      <td>0.003486</td>
      <td>0.000810</td>
      <td>1.675904e-05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mom_ed1</td>
      <td>alpha_1</td>
      <td>-0.002602</td>
      <td>0.004000</td>
      <td>5.153889e-01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mom_ed1</td>
      <td>alpha_2</td>
      <td>0.000035</td>
      <td>0.000059</td>
      <td>5.589726e-01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mom_ed1</td>
      <td>alpha_3</td>
      <td>-0.000123</td>
      <td>0.000054</td>
      <td>2.255634e-02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gest</td>
      <td>alpha_1</td>
      <td>-0.128477</td>
      <td>0.030871</td>
      <td>3.160613e-05</td>
    </tr>
    <tr>
      <th>10</th>
      <td>gest</td>
      <td>alpha_2</td>
      <td>0.005786</td>
      <td>0.000457</td>
      <td>1.068860e-36</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gest</td>
      <td>alpha_3</td>
      <td>0.004670</td>
      <td>0.000416</td>
      <td>3.529121e-29</td>
    </tr>
    <tr>
      <th>13</th>
      <td>nprenatal</td>
      <td>alpha_1</td>
      <td>0.089424</td>
      <td>0.054626</td>
      <td>1.016293e-01</td>
    </tr>
    <tr>
      <th>14</th>
      <td>nprenatal</td>
      <td>alpha_2</td>
      <td>0.003341</td>
      <td>0.000808</td>
      <td>3.517186e-05</td>
    </tr>
    <tr>
      <th>15</th>
      <td>nprenatal</td>
      <td>alpha_3</td>
      <td>0.003797</td>
      <td>0.000739</td>
      <td>2.778929e-07</td>
    </tr>
    <tr>
      <th>17</th>
      <td>yob</td>
      <td>alpha_1</td>
      <td>0.589951</td>
      <td>0.056977</td>
      <td>4.065631e-25</td>
    </tr>
    <tr>
      <th>18</th>
      <td>yob</td>
      <td>alpha_2</td>
      <td>0.005239</td>
      <td>0.000844</td>
      <td>5.297584e-10</td>
    </tr>
    <tr>
      <th>19</th>
      <td>yob</td>
      <td>alpha_3</td>
      <td>0.008469</td>
      <td>0.000768</td>
      <td>2.960432e-28</td>
    </tr>
  </tbody>
</table>
</div>



### A Side Note on the Caliper

In the `run_rdd()` function above, there is a `caliper` argument. The caliper is determined by the analyst, and it refers to the segment of the sample in which we limit our analysis. In the example above, the caliper is 85, meaning we'll look at newborns with birth weights 85 grams above the threshold and 85 grams below the threshold inclusive. 

Intuitively, units that lie arbitrarily close to the caliper will be much more similar than units that lie arbitrarily far from the caliper. Smaller calipers allow us to better identify treatment effects because treatment and control groups resemble one another. 

With a narrow caliper, we trade-off low bias for high variance. A more narrow caliper means a smaller sample size. As n decreases our standard errors increase - with less data to work with we are less certain of our results. 

On the other hand, a wider caliper results in a high bias, low variance trade-off. A wider caliper allows us to draw from a larger subset of the sample. As n increases we can be more certain of our estimates. 

Later on in the analysis, we'll also see how this trade-off in action when experimenting with different sized calipers.

### Synthesizing the Test for Smoothness

Three out of the five background covariates (`mom_age`,`gest`, and `yob`) have coefficients for `alpha_1`, the the size of the discontinuity, that are statistically significant at the 95% confidence interval which suggests that there has been manipulation of these background covariates. According to the authors' [online appendix](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2903901/bin/NIHMS180487-supplement-supplement_1.pdf), the estimates and standard errors are consistent with my calculations. In their paper, they acknowledge the statistically meaningful differences, but conclude that the smoothness assumption is upheld in spite of the p-values. A case can be made that none of these differences are practically meaningful, however. For example, on average, there may not be a difference between a mother who is 26.44 years old and a mother who is 26.39 years old.

As a reminder, if there are discontinuities through the threshold, then our smoothness assumption will be violated. Without the smoothness assumption, then our treatment and control groups may not be balanced and, consequently, RDD will not estimate a local average treatment effect.

# Estimating the Discontinuity in Mortality Rates (Without Covariates)

The section above measured the size of the discontinuity with the background covariates as the outcome variable. In this section, we use an identical right-hand side expression, but now the outcome variables are the two measures of mortality rates. We estimate the discontinuity using the regression:

$Y_i = a_0 + a_1VLBW + a_2VLBW_i(g_i - 1500) + a_3(1 - VLBW_i)(g_i - 1500) + e_i$

where $Y_i$ is either the one-year mortality rate or the 28-day mortality rate.


```python
run_rdd(dataframe=df,
        dep_vars=['agedth5', 'agedth4'],
        ind_vars=['alpha_1','alpha_2', 'alpha_3'],
        caliper=85)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENDOGENOUS_VARIABLE</th>
      <th>EXOGENOUS_VARIABLE</th>
      <th>ESTIMATE</th>
      <th>STD_ERROR</th>
      <th>P_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>agedth5</td>
      <td>alpha_1</td>
      <td>-0.009510</td>
      <td>0.002153</td>
      <td>1.002399e-05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>agedth5</td>
      <td>alpha_2</td>
      <td>-0.000135</td>
      <td>0.000032</td>
      <td>2.261106e-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>agedth5</td>
      <td>alpha_3</td>
      <td>-0.000225</td>
      <td>0.000029</td>
      <td>9.640730e-15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>agedth4</td>
      <td>alpha_1</td>
      <td>-0.008781</td>
      <td>0.001811</td>
      <td>1.250702e-06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>agedth4</td>
      <td>alpha_2</td>
      <td>-0.000113</td>
      <td>0.000027</td>
      <td>2.478069e-05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>agedth4</td>
      <td>alpha_3</td>
      <td>-0.000200</td>
      <td>0.000024</td>
      <td>2.904020e-16</td>
    </tr>
  </tbody>
</table>
</div>



Note that `agedth5` corresponds to the one-year mortality rate. `agedth4` corresponds to the 28-day mortality rate.

`alpha_1` estimates the local average treatment effect for newborns in the vicinity of the threshold. When looking at the one-year mortality rate, the coefficient is -.0095. Newborns born just below the threshold have about a one percentage point lower mortality rate than newborns born just above the cutoff of 1500 grams. The 28-day mortality rate is similar as there is a there is about a nine-tenths of a percentage point reduction (-.00878) in the mortality rate for newborns just under 1500 grams. Both estimates are statistically significant at the 95% confidence interval. 

As the researchers discuss, the reduction in mortality rate for infants just below the threshold may be attributed to the additional level of care that these newborns receive because of their very low birth weight categorization. While 1% may seem trivially small, within this context it is substantively fairly large. Our mortality rates are roughly 4% to 6% in the neighborhood of the threshold so 1% is a fairly "large" impact in this context.

`alpha_2` estimates the slope of the line for mortality rates for newborns categorized as very low birth weight. Its negative value (-.000135 for one-year mortality rate and -.000113 for 28-day mortality rate) informs us that as we approach the threshold from below, the mortality rate decreases. In other words, newborns born with a weight closer to 0 grams will be more at risk than those born further away from 0 grams. Even with increased medical treatment, it likely becomes more difficult to save a newborn who is further away from the threshold in the negative direction.

`alpha_3` estimates the slope of the line for mortality rates for newborns not categorized as very low birth weight. Its negative value (-.0002248 for one-year mortality rate and -.0001997 for 28-day mortality rate) suggest that the mortality rate decreases as birth weight increases for newborns weighing more than 1500 grams at birth. This makes sense given that babies who weigh more at birth tend to be healthier and less at risk compared to babies who weigh less at birth.

# Estimating the Discontinuity in Mortality Rates (With Covariates)

In this section I add in covariates. Recall from the section where background covariates are plotted - there was suggestive evidence of discontinuities when plotting them against birth weight bins when looking at p-values. Typically, if our RDD assumptions hold then there is no need to add covariates because our two groups are balanced on average. I add in mother's age, indicators for mother's education and race, indicators for year of birth, and indicators for gestational age and prenatal care visits into the model and re-examine the coefficient for `alpha_1`.


```python
dummies_df = pd.get_dummies(df, columns=['yob', 'mom_race'])

covariates = [
    'alpha_1',        'alpha_2',         'alpha_3',        'mom_age',
    'mom_ed1',        'mom_ed2',         'mom_ed3',        'mom_ed4',
    'mom_race_white', 'mom_race_black',  'mom_race_other', 'yob_1984',    
    'yob_1985',       'yob_1986',        'yob_1987',       'yob_1988',    
    'yob_1989',       'yob_1990',        'yob_1991',       'yob_1995',    
    'yob_1996',       'yob_1997',        'yob_1998',       'yob_1999',    
    'yob_2000',       'yob_2001',        'yob_2002',       'gest_wks1',   
    'gest_wks2',      'gest_wks3',       'nprenatal_1',    'nprenatal_2', 
    'nprenatal_3'
]

run_rdd(dataframe=dummies_df,
        dep_vars=['agedth5', 'agedth4'],
        ind_vars=covariates,
        caliper=85)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENDOGENOUS_VARIABLE</th>
      <th>EXOGENOUS_VARIABLE</th>
      <th>ESTIMATE</th>
      <th>STD_ERROR</th>
      <th>P_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>agedth5</td>
      <td>alpha_1</td>
      <td>-0.007664</td>
      <td>0.002144</td>
      <td>3.513064e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>agedth5</td>
      <td>alpha_2</td>
      <td>-0.000123</td>
      <td>0.000032</td>
      <td>1.103700e-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>agedth5</td>
      <td>alpha_3</td>
      <td>-0.000201</td>
      <td>0.000029</td>
      <td>4.032542e-12</td>
    </tr>
    <tr>
      <th>35</th>
      <td>agedth4</td>
      <td>alpha_1</td>
      <td>-0.007443</td>
      <td>0.001805</td>
      <td>3.728451e-05</td>
    </tr>
    <tr>
      <th>36</th>
      <td>agedth4</td>
      <td>alpha_2</td>
      <td>-0.000104</td>
      <td>0.000027</td>
      <td>1.007497e-04</td>
    </tr>
    <tr>
      <th>37</th>
      <td>agedth4</td>
      <td>alpha_3</td>
      <td>-0.000183</td>
      <td>0.000024</td>
      <td>5.809430e-14</td>
    </tr>
  </tbody>
</table>
</div>



The new estimates are slightly diminished. The one-year mortality rate decrease from -0.009510 to -0.007664 in absolute terms, and the 28-day mortality rate decreases from -0.008781 to -0.007443 in absolute terms. Since the new estimates are smaller than before, this suggests bias in our initial estimates.

# The Effects of the Caliper

Next, I examine what happens to the estimates when changing the size of the caliper. I reduce the caliper from 85 grams to 30 grams, and then increase it from 85 grams to 120 grams.

#### Narrowing the Caliper


```python
run_rdd(dataframe=dummies_df,
        dep_vars=['agedth5', 'agedth4'],
        ind_vars=covariates,
        caliper=30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENDOGENOUS_VARIABLE</th>
      <th>EXOGENOUS_VARIABLE</th>
      <th>ESTIMATE</th>
      <th>STD_ERROR</th>
      <th>P_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>agedth5</td>
      <td>alpha_1</td>
      <td>-0.014260</td>
      <td>0.005174</td>
      <td>5.851169e-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>agedth5</td>
      <td>alpha_2</td>
      <td>-0.000252</td>
      <td>0.000205</td>
      <td>2.181580e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>agedth5</td>
      <td>alpha_3</td>
      <td>-0.000641</td>
      <td>0.000132</td>
      <td>1.195078e-06</td>
    </tr>
    <tr>
      <th>35</th>
      <td>agedth4</td>
      <td>alpha_1</td>
      <td>-0.014402</td>
      <td>0.004375</td>
      <td>9.956681e-04</td>
    </tr>
    <tr>
      <th>36</th>
      <td>agedth4</td>
      <td>alpha_2</td>
      <td>-0.000277</td>
      <td>0.000173</td>
      <td>1.089350e-01</td>
    </tr>
    <tr>
      <th>37</th>
      <td>agedth4</td>
      <td>alpha_3</td>
      <td>-0.000551</td>
      <td>0.000112</td>
      <td>7.827446e-07</td>
    </tr>
  </tbody>
</table>
</div>



By shrinking the caliper, the estimated size of the discontinuity nearly *doubles* from -0.007664 to -0.014260 when looking at the one-year mortality rate, and the size of the effect is consistent when looking at the 28-day mortality rate as the effect increases from -0.007443 to -0.014402. 

Simultaneously, the standard errors also increase by at least two-fold. In the one-year mortality rate, the standard error jumps from 0.002144 to 0.005174, and in the 28-day mortality rate, the standard error jumps from 0.001805 to 0.004375.

#### Expanding the Caliper


```python
run_rdd(dataframe=dummies_df,
        dep_vars=['agedth5', 'agedth4'],
        ind_vars=covariates,
        caliper=120)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENDOGENOUS_VARIABLE</th>
      <th>EXOGENOUS_VARIABLE</th>
      <th>ESTIMATE</th>
      <th>STD_ERROR</th>
      <th>P_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>agedth5</td>
      <td>alpha_1</td>
      <td>-0.006436</td>
      <td>0.001761</td>
      <td>2.569347e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>agedth5</td>
      <td>alpha_2</td>
      <td>-0.000141</td>
      <td>0.000020</td>
      <td>1.134444e-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>agedth5</td>
      <td>alpha_3</td>
      <td>-0.000120</td>
      <td>0.000014</td>
      <td>1.257010e-17</td>
    </tr>
    <tr>
      <th>35</th>
      <td>agedth4</td>
      <td>alpha_1</td>
      <td>-0.005646</td>
      <td>0.001483</td>
      <td>1.405483e-04</td>
    </tr>
    <tr>
      <th>36</th>
      <td>agedth4</td>
      <td>alpha_2</td>
      <td>-0.000113</td>
      <td>0.000017</td>
      <td>1.289948e-11</td>
    </tr>
    <tr>
      <th>37</th>
      <td>agedth4</td>
      <td>alpha_3</td>
      <td>-0.000097</td>
      <td>0.000012</td>
      <td>1.716793e-16</td>
    </tr>
  </tbody>
</table>
</div>



In expanding the caliper, the the opposite effect occurs. The estimated effect shrinks from -0.007664 to -0.006436 when looking at one-year mortality rates. Similarly, in the 28-day mortality rate, the effect falls slightly from -0.007443 to -0.005646.

As expected, the standard errors also tick down from 0.002144 to 0.001761 and from 0.001805 to 0.001483 in analyzing the one-year mortality rate and 28-day mortality rate respectively.

#### The Bias Variance Trade-Off

In sum, by changing the caliper, we trade-off bias for variance. When the caliper shrinks, we have less data to work with, so our estimates become less bias but variance increases since we have less data to explain variation. When the caliper grows, we have more data to work with, so our estimates become more bias but variance decreases since we have more data points to look towards in explaining variation.

# Key Takeaways 

## Research Design

Regression Discontinuity Design serves as a good alternative to randomized control trials where randomization is not ethically or practically possible. RDD takes advantage of a threshold on a running variable. Observations on either side of the threshold are hypothetically similar enough, which mimics the balance typically achieved (but not always guaranteed) through randomization. Any differences in outcomes, then, can be attributed to treatments and not differences due to the differing background characteristics of the two groups. 

## Do the Benefits Outweigh the Costs?

In returning to the broader question proposed originally, do the benefits of added medical attention outweigh its expenditures? The study provides insight into answering the question, but we'll need more information for a more thorough cost-benefit analysis.

1. *The costs*: This includes the initial cost of additional medical services received for a newborn being classified as very low birth weight, and the downstream costs that occur over the course of this newborn’s life. Medical technology and compensation for nurses and doctors fall under the upfront costs while downstream fees include care related to physical and developmental complications that arise from being born with a low birth weight. Additionally, there may be benefit payments provided by a social safety net program. Finally, we would also need to consider the normal cost of maintaining a person’s life over time.

2. *The benefits*: It’s difficult to assign a dollar amount to a person’s life, but that has not stopped people (Viscusi 1993, Cutler and Richardson 1998, and Cutler and Meara 1999) from authoring research papers on the topic. The literature suggests that the benefits can be measured on several dimensions: the value a person adds to society, the value of what they earn over their lifetimes, or the value of what they contribute less what they consume. Alternatively, researchers suggest asking people how much they value life via survey research. However, the consensus throughout the literature recommends using a compensating wage differentials framework, suggested by Adam Smith in the 1700s. The frameworks calculates the statistical value of a human life by inferring how much people have to be paid to work risky jobs, or alternatively, asking how much they would pay to avoid a small risk to their lives? Researchers suggest that the dollar amount to this calculation is \\$75,000 to \\$150,000 per year or roughly \\$3 million to \\$7 million for a middle aged person.

3. *The discount rate*: This is subject to analyst judgment because some would argue that the rate should be 0 as life, in principle, should not be discounted. Conversely, others suggest 20%+ to justify risky behaviors.

While a lot of this information is subjective and even unattainable, if we follow Almond’s line of thought, (which does not include all the considerations mentioned above) then the returns to additional medical treatment for very low birth weight newborns born around the 1500 gram threshold are *tremendous*. Almond et al. estimate that the additional cost incurred for being just below the threshold is \\$3,795, which is then divided by the reduction in one-year mortality rate from above the threshold in absolute terms (.0072) This approximates a cost of \\$527,083 per newborn. Using the \\$3 million measure of benefit, this yields a roughly 470\% return without discounting.

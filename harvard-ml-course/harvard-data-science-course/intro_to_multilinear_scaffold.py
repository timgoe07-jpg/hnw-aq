# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi

# Read the file "Advertising.csv"
df = pd.read_csv('Advertising.csv')

# Take a quick look at the dataframe
df.head()

# Define an empty Pandas dataframe to store the R-squared value associated with each 
# predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])

# For each predictor in the dataframe, call the function "fit_and_plot_linear()"
# from the helper file with the predictor as a parameter to the function

# This function will split the data into train and test split, fit a linear model
# on the train data and compute the R-squared value on both the train and test data

# **Your code here**
for predictor in ['TV', 'Radio', 'Newspaper']:
    r2_train, r2_test = fit_and_plot_linear(df[[predictor]])
    df_results.loc[len(df_results)] = {
        'Predictor': predictor,
        'R2 Train': r2_train,
        'R2 Test': r2_test
    }

### edTest(test_chow1) ###
# Submit an answer choice as a string below 
# (Eg. if you choose option C, put 'C')
# Based on the plot and results, which model do you think is the best for prediction?

# A. TV
# B. Radio
# C. Newspaper
# D. Sales

answer1 = 'A'
# strongest linear relationship with sales, tightest scatter around fitted line and highest R2

# Call the function "fit_and_plot_multi()" from the helper to fit a multilinear model
# on the train data and compute the R-squared value on both the train and test data

# **Your code here**
r2_train_multi, r2_test_multi = fit_and_plot_multi()

### edTest(test_dataframe) ###

# Store the R-squared values for all models
# in the dataframe intialized above
# **Your code here**

df_results.loc[len(df_results)] = {
    'Predictor': 'multilinear',
    'R2 Train': r2_train_multi,
    'R2 Test': r2_test_multi
}

# Take a quick look at the dataframe
df_results.head()

# ⏸ Why do you think the mutilinear regression model is better?

# A. The model goes to the gym thrice as more when compare to the linear model.
# B. The model has more information from various predictors/features.
# C. The model is not linear, hence fits the complex data.
# D. The model is not better than the simple linear regression.

### edTest(test_chow2) ###
# Submit an answer choice as a string below 
# (Eg. if you choose option C, put 'C')
answer2 = 'B'
# Highest R2 on train = 0.9067, highest R2 on test = 0.8601 
# => sig. improvement over best single predictor (TV, 0.6763 test R2)

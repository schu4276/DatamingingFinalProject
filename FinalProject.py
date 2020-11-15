# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:02:40 2020

@author: Cassie Aaron Utkarsh
"""

import pandas as pd
import matplotlib.pylab as plt

# Defining dataset and exploring data
UFO_df = pd.read_csv('/DataMining/lab2/complete.csv')
low_memory=False
# Cleaning the data to have a cleaned data set for the rest of the project
# Cleaning data, there are quite a few pieces missing from the data set, so this 
# Needs to be addressed

UFO_cleaned = UFO_df.drop(columns = ['duration (hours/min)', 'Unnamed: 11','country','comments','date posted','datetime'])
UFO_cleaned = UFO_cleaned.dropna() # droping columns w/ NA values

# changing duration to numerical value
UFO_cleaned = UFO_cleaned.rename(columns={'duration (seconds)': 'duration'}) 
UFO_cleaned.duration =  UFO_cleaned.duration.astype('float64')

UFO_cleaned.latitude
# getting rid of values that that were messed up/ invalid 
UFO_cleaned = UFO_cleaned[~UFO_cleaned.latitude.str.contains('/')]
UFO_cleaned = UFO_cleaned[~UFO_cleaned.latitude.str.contains('q')]

UFO_cleaned.latitude = UFO_cleaned.latitude.astype('float64')


UFO_cleaned = pd.get_dummies(UFO_cleaned, columns=['shape','state'], prefix_sep='_') 
 
UFO_cleaned.info()

#  Correlation table for quantative cereal data
import seaborn as sns;  # had to add this import
## simple heatmap of correlations (without values)
corr = UFO_cleaned.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# Change the colormap to a divergent scale and fix the range of the colormap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap= "RdBu")
# Include information about values (example demonstrate how to control the size of
# the plot
fig, ax = plt.subplots()
fig.set_size_inches(18, 18)
sns.heatmap(corr, annot=False, fmt= ".1f", cmap= "RdBu", center=0, ax=ax)
# darker and bluer means stronger correlation
# used to visualize correlations and missing values


## Multiple Linear Regression 
UFO_cleaned_loc = UFO_cleaned.iloc[0:1000]
UFO_cleaned_loc.info()

predictors = UFO_cleaned_loc.drop(columns=['duration', 'city'])
print(predictors)
outcome= UFO_cleaned_loc.duration

from sklearn.model_selection import train_test_split

X = predictors 
y = UFO_cleaned_loc['duration']
X.shape
y.shape
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.4, random_state=1)

from sklearn.linear_model import LinearRegression
UFO_lm = LinearRegression()
UFO_lm.fit(train_X, train_y)

# print coefficiient
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': UFO_lm.coef_}))

from dmba import regressionSummary
regressionSummary(train_y, UFO_lm.predict(train_X))

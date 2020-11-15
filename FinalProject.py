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

US_df = UFO_df.drop(columns = ['duration (hours/min)', 'Unnamed: 11','comments','date posted','datetime'])

US_df.country = US_df[US_df.country == 'us']   


US_df.country.value_counts()
US_df = US_df.dropna() # droping columns w/ NA values

# changing duration to numerical value
US_df = US_df.rename(columns={'duration (seconds)': 'duration'}) 
US_df.duration =  US_df.duration.astype('float64')

US_df.latitude
# getting rid of values that that were messed up/ invalid 
US_df = US_df[~US_df.latitude.str.contains('/')]
US_df = US_df[~US_df.latitude.str.contains('q')]

US_df.latitude = US_df.latitude.astype('float64')
 
US_df.info()



US_df = pd.get_dummies(US_df, columns=['shape'], prefix_sep='_') 
US_df['combined'] = list(zip(US_df.latitude, US_df.longitude))

## Correlation table for quantative cereal data
import seaborn as sns;  # had to add this import
# simple heatmap of correlations (without values)
corr = US_df.corr()
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
plt.show()


## Multiple Linear Regression 
US_df_loc = US_df.iloc[0:1000]
US_df_loc.info()
unwanted = ['city','country', 'latitude', 'longitude','state', 'combined']

print(predictors)
outcome = ['latitude', 'longitude']


from sklearn.model_selection import train_test_split

X = US_df_loc.drop(columns = unwanted) 
y = US_df_loc['latitude']
X.shape
y.shape
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.4, random_state=1)

from sklearn.linear_model import LinearRegression
UFO_lm = LinearRegression()
UFO_lm.fit(train_X, train_y)

# print coefficiient
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': UFO_lm.coef_}))
US_df_loc

from dmba import regressionSummary
regressionSummary(train_y, UFO_lm.predict(train_X))

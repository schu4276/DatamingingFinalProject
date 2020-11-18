# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:02:40 2020

@author: Cassie Aaron Utkarsh
"""

import pandas as pd

# Defining dataset and exploring data
UFO_df = pd.read_csv(r'UFO/complete.csv', low_memory=False)

# To see all columns
pd.set_option('display.max_columns', None)

UFO_df.info()

### Cleaning the data to have a cleaned data set for the rest of the project ###

# These are the rows where the unnamed column has values. This unnamed columns exists
# because all 195 of these rows have data in the wrong columns or missing data.
badRows = UFO_df[UFO_df['Unnamed: 11'].notnull()]
print(badRows)

# Drop those rows by passing their index to .drop() and using axis=0.
# Also drop the Unnamed column which is now full of NA values
UFO_df.drop(badRows.index, axis=0, inplace=True)
UFO_df.drop(columns=['Unnamed: 11'], inplace=True)

# Drop all rows with NA values
UFO_df.dropna(inplace=True)
# Rename duration to remove unit. Rename shape, so it doesn't clash with DataFrame.shape method
UFO_df.rename(columns={'duration (seconds)': 'duration', 'shape': 'UFO_shape'}, inplace=True)

# Convert data type of duration and latitude columns from object to float64 
UFO_df['duration'] = pd.to_numeric(UFO_df.duration)
UFO_df['latitude'] = pd.to_numeric(UFO_df.latitude)

UFO_df.info()

# We are removing datetime for now, but really we might use it later.
# Drop the duration column given in hours/min, since there is a column of duration in seconds.
# We are not interested in the date sightings were posted online or the comments
# associated with those sightings.
UFO_df_cleaned = UFO_df.drop(columns=['datetime', 'duration (hours/min)', 'comments', 'date posted'])


######### Focusing on US UFO Sighting #########

# New Dataframe with just sighting in United States.
US_df = UFO_df_cleaned[UFO_df_cleaned.country == 'us']   

# Confirming the country column only has United States sightings.
US_df.country.value_counts()

# Turn shapes into dummy variables
US_df = pd.get_dummies(US_df, columns=['UFO_shape'], prefix_sep='_') 
# Create a new column that combines latitude and longitude
US_df['combined'] = list(zip(US_df.latitude, US_df.longitude))


######### Correlation Table and Visualization #########

import seaborn as sns
import matplotlib.pylab as plt


corr = US_df.corr() # correlation table
# simple heatmap of correlations (without values)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# Change the colormap to a divergent scale and fix the range of the colormap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap= "RdBu")

# Include information about values 
fig, ax = plt.subplots()
fig.set_size_inches(18, 18)
sns.heatmap(corr, annot=False, fmt= ".1f", cmap= "RdBu", center=0, ax=ax)
# darker and bluer means stronger correlation
# used to visualize correlations and missing values
plt.show()


######### Partitioning Data for Stepwise Regression #########
from sklearn.model_selection import train_test_split

US_df_loc = US_df.iloc[0:1000]
US_df_loc.info()
unwanted = ['city','country', 'latitude', 'longitude','state', 'combined']

X = US_df_loc.drop(columns = unwanted) 
y = US_df_loc['longitude']
X.shape
y.shape
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.4, random_state=1)
# print coefficients




# print performance measures (training data)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.metrics import roc_curve, auc
import statsmodels.formula.api as sm
import math

from dmba import regressionSummary, exhaustive_search, classificationSummary
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba import liftChart, gainsChart


US_lm = LinearRegression()
US_lm .fit(train_X, train_y)

regressionSummary(train_y, US_lm.predict(train_X))

def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(train_X[variables], train_y)
    return model

def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(train_y, [train_y.mean()] * len(train_y), model, df=1)
    return AIC_score(train_y, model.predict(train_X[variables]), model)


best_model, best_variables = stepwise_selection(train_X.columns, train_model, score_model, 
                                 verbose=True)
print(best_variables)
regressionSummary(valid_y, best_model.predict(valid_X[best_variables]))
# lift chart for stepwise model

pred_v = pd.Series(best_model.predict(valid_X[best_variables]))
pred_v = pred_v.sort_values(ascending=False)

fig, axes = plt.subplots(nrows=1, ncols=2)
ax = gainsChart(pred_v, ax=axes[0])
ax.set_ylabel('Cumulative Price')
ax.set_title('Cumulative Gains Chart')

ax = liftChart(pred_v, ax=axes[1], labelBars=False)
ax.set_ylabel('Lift')

plt.tight_layout()
plt.show()


## Multiple Linear Regression With Seperate Lat and Long
US_df_loc = US_df.iloc[0:1000]


#outcome = ['latitude']
outcome = ['longitude']


X = US_df_loc[best_variables]
y = US_df_loc['latitude']
X.shape
y.shape
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.4, random_state=1)

UFO_lm = LinearRegression()
UFO_lm.fit(train_X, train_y)

# print coefficiient
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': UFO_lm.coef_}))
US_df_loc

regressionSummary(train_y, UFO_lm.predict(train_X))

## Using Nearest Neighbor
US_df.shape_changing
# print coefficients
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# Transform the full dataset
UFO_df = pd.read_csv('/DataMining/lab2/complete.csv',low_memory=False)
trainData, validData = train_test_split(US_df, test_size = 0.4, random_state = 1)

scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['shape_changing', 'duration']])

# Transforming dataset
Norm = pd.concat([pd.DataFrame(scaler.transform(US_df[['shape_changing', 'duration']]), 
                               columns=['duration','shape_changing']), 
                  US_df[['latitude', 'longitude']]], axis=1)
trainData.index

trainNorm = Norm.iloc[trainData.index]
validNorm = Norm.iloc[validData.index]

newUFO = pd.DataFrame([{'duration': 300, 'shape_changing': 1}])

newUFONorm = pd.DataFrame(scaler.transform(newUFO),columns=['duration','shape_changing'])


# Use k-nearest neighbor
newUFO = pd.DataFrame([{'duration': 30, 'shape_changing':0 }])
knn = NearestNeighbors(n_neighbors=3)
knn.fir(trainNorm[['duration', 'shape_changing']])
distances, indicies = knn.kneighbors(newUFONorm)
print(trainNorm.iloc[indicies[0], :])


train_X = trainNorm[['shape_cigar','shape_disk','shape_circle','shape_changing', 'duration']]
train_y = trainNorm['latiude']
valid_X = validNorm[['shape_cigar', 'shape_disk','shape_circle','shape_changing','duration']]
valid_y = validNorm['latiude']

# Train a classifier for different values of K
results = []
for k in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
    results.append({
        'k':k,
        'accuracy': accuracy_score(valid_y, knn.predict(valid_X))
    })
# Convert results to a pandas df
results = pd.DataFrame(results)
print(results)
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:02:40 2020
@author: Cassie Aaron Utkarsh
"""

############## Imports ##############
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from wordcloud import WordCloud, STOPWORDS

from dmba import regressionSummary, stepwise_selection, AIC_score
from dmba import liftChart, gainsChart, classificationSummary

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
#####################################################

# Defining dataset and exploring data
UFO_df = pd.read_csv(r'UFO/complete.csv',low_memory=False)

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


# We are removing datetime for now, but really we might use it later.
# Drop the duration column given in hours/min, since there is a column of duration in seconds.
# We are not interested in the date sightings were posted online or the comments
# associated with those sightings.
UFO_df_cleaned = UFO_df.drop(columns=['datetime', 'duration (hours/min)', 'comments', 'date posted'])


##################### Focusing on UFO Sighting in USA #####################

# New Dataframe with just sighting in United States.
US_df = UFO_df_cleaned[UFO_df_cleaned.country == 'us']   

# Confirming the country column only has United States sightings.
US_df.country.value_counts()

# Turn shapes into dummy variables
US_df = pd.get_dummies(US_df, columns=['UFO_shape'], prefix_sep='_') 
# Create a new column that combines latitude and longitude
US_df.iloc[5]
US_df['combined'] = list((US_df.latitude - US_df.longitude))
US_df.combined

######### Correlation #########

corr = US_df.corr() # correlation table
# simple heatmap of correlations (without values)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# Change the colormap to a divergent scale and fix the range of the colormap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap= "RdBu")

# Include information about values 
fig, ax = plt.subplots()
fig.set_size_inches(12, 12)
sns.heatmap(corr, annot=False, fmt= ".1f", cmap= "RdBu", center=0, ax=ax)
# darker and bluer means stronger correlation
# used to visualize correlations and missing values
plt.show()


######### Partitioning Data for Stepwise Regression #########

US_df_loc = US_df.iloc[0:1000]
US_df_loc.info()
unwanted = ['city','country', 'latitude', 'longitude','state', 'combined']


X = US_df_loc.drop(columns = unwanted).astype(float) 
X.info()
y = US_df_loc['combined']
X.shape
y.shape
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.4, random_state=1)
# print coefficients


# print performance measures (training data)
US_lm = LinearRegression()
US_lm .fit(train_X, train_y)


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


## Multiple Linear Regression With Combined Lat and Long
US_df_loc = US_df.iloc[0:1000]

#outcome = ['latitude']
outcome = ['combined']
X = US_df_loc[best_variables]
y = US_df_loc['combined']
X.shape
y.shape
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.4, random_state=1)

UFO_lm = LinearRegression()
UFO_lm.fit(train_X, train_y)

# print coefficiient
regressionSummary(train_y, UFO_lm.predict(train_X))

## Using Nearest Neighbor

# Transform the full dataset
US_df['Number'] = US_df.index+1
US_df
trainData, validData = train_test_split(US_df, test_size = 0.4, random_state = 1)

scaler = preprocessing.StandardScaler()
scaler.fit(trainData[best_variables])

# Transforming dataset
Norm = pd.concat([pd.DataFrame(scaler.transform(US_df[best_variables]), 
                               columns=['zUFO_shape_fireball','zUFO_shape_disk', 'zUFO_shape_circle','zUFO_shape_teardrop']), 
                  US_df[['combined','Number']]], axis=1)
trainData.index
Norm.index
trainNorm = Norm.iloc[trainData.index/100]
validNorm = Norm.iloc[validData.index/100]

newUFO = pd.DataFrame([{'UFO_shape_fireball': 0,
 'UFO_shape_disk':0,
 'UFO_shape_circle':1,
 'UFO_shape_teardrop':0}])

newUFONorm = pd.DataFrame(scaler.transform(newUFO),columns=[best_variables])


# Use k-nearest neighbor
Norm.dropna()
knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zUFO_shape_fireball','zUFO_shape_disk','zUFO_shape_circle','zUFO_shape_teardrop']])
distances, indicies = knn.kneighbors(newUFONorm)
print(trainNorm.iloc[indicies[0], :])

trainNorm = trainNorm.dropna()
train_X = trainNorm[['zUFO_shape_fireball','zUFO_shape_disk','zUFO_shape_circle','zUFO_shape_teardrop']]
train_y = trainNorm['combined'].astype(int)
validNorm = validNorm.dropna()
valid_X = validNorm[['zUFO_shape_fireball','zUFO_shape_disk','zUFO_shape_circle','zUFO_shape_teardrop']]
valid_y = validNorm['combined'].astype(int)


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

#### Creating Supporting Visuals


## UFO sightings in United States by State (not including Puerto Rico or DC)
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go

US_df2 = US_df[US_df['state'] != 'pr']
US_df2 = US_df2[US_df2['state'] != 'dc']

unique_states = US_df2.groupby('state').state.count().index
unique_states = [x.upper() for x in unique_states]

ufo_bystate = US_df2.groupby('state').state.count().values


fig = go.Figure(data=go.Choropleth(
    locations = unique_states, # Spatial coordinates
    z = ufo_bystate, # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "UFO sighting",
))

fig.update_layout(
    title_text = 'UFO sightings in US by states',
    geo_scope='usa', # limite map scope to USA
)

fig.show()

# UFO sightings per 100,000 people in state
state_population = np.asarray([738432, 4858979, 2978204, 6828065, 39144818, 5456574,
                               3590886, 945934, 20271272, 10214860, 1431603,
                               3123899, 1654930, 12859995, 6619680, 2911641, 4425092,
                               4670724, 6794422, 6006401, 1329328, 9922576, 5489594,
                               6083672, 2992333, 1032949, 10042802, 756927, 1896190,
                               1330608, 8958013, 2085109, 2890845, 19795791, 11613423,
                               3911338, 4028977, 12802503, 1056298, 4896146, 858469,
                               6600299, 27469114, 2995919, 8382993, 626042, 7170351,
                               5771337, 1844128, 586107])


ufo_percapita = np.round(ufo_bystate / state_population * 100000, 2)

fig = go.Figure(data=go.Choropleth(
    locations = unique_states, # Spatial coordinates
    z = ufo_percapita, # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "UFO sighting per capita",
))

fig.update_layout(
    title_text = 'UFO sightings in US per capita',
    geo_scope='usa', # limits map scope to USA
)

fig.show()

###### Word Cloud 

# Start with one review:
text = " ".join(comments for comments in UFO_df.comments)
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


###### Naive Bayes 

predictors = ['UFO_shape_changed', 'UFO_shape_changing', 'UFO_shape_chevron',
       'UFO_shape_cigar', 'UFO_shape_circle', 'UFO_shape_cone',
       'UFO_shape_crescent', 'UFO_shape_cross', 'UFO_shape_cylinder',
       'UFO_shape_delta', 'UFO_shape_diamond', 'UFO_shape_disk',
       'UFO_shape_egg', 'UFO_shape_fireball', 'UFO_shape_flare',
       'UFO_shape_flash', 'UFO_shape_formation', 'UFO_shape_hexagon',
       'UFO_shape_light', 'UFO_shape_other', 'UFO_shape_oval',
       'UFO_shape_pyramid', 'UFO_shape_rectangle', 'UFO_shape_round',
       'UFO_shape_sphere', 'UFO_shape_teardrop', 'UFO_shape_triangle',
       'UFO_shape_unknown', 'duration']
outcome = 'state'

NB_x = US_df[predictors]
NB_y = US_df['state'].astype('category')
classes = list(NB_y.cat.categories)

# split into training and validation sets 
x_train, x_valid, y_train, y_valid = train_test_split(NB_x, NB_y, test_size=0.40, random_state=1)

# now running the modeling 
delays_nb = MultinomialNB(alpha= 0.01)
delays_nb.fit(x_train, y_train)

# predict probabilities
predProb_train = delays_nb.predict_proba(x_train)
predProb_valid = delays_nb.predict_proba(x_valid)

# predicting the class membership
y_valid_pred = delays_nb.predict(x_valid)

# predicting the class membership
y_train_pred = delays_nb.predict(x_train)

# training confusion matrix 
classificationSummary(y_train, y_train_pred, class_names=classes) 

# validation confusion matrix 
classificationSummary(y_valid, y_valid_pred, class_names=classes) 

df = pd.DataFrame({'actual': 1 - y_valid.cat.codes, 'prob': predProb_valid[:, 0]})
df = df.sort_values(by=['prob'], ascending=False).reset_index(drop=True)

fig, ax = plt.subplots()
fig.set_size_inches(4, 4)
gainsChart(df.actual, ax=ax)


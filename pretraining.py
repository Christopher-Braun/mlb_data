import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from numpy import linalg
from numpy.linalg import norm
import xgboost
import tensorflow as tf

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])

from constant_variables import features_top_list
X2 = X[features_top_list]
X3 = X2.drop('Expected_Runs', axis = 1)

#X_train, X_validate, X_test = np.split(X.sample(frac=1), [int(.6*len(X)), int(.8*len(X))])
#y_train, y_validate, y_test = np.split(y.sample(frac=1), [int(.6*len(y)), int(.8*len(y))])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size = 0.20, random_state = 0)

from sklearn.model_selection import train_test_split
X_train, X_train1, y_train, y_train1 = train_test_split(X_train, y_train, test_size = 0.50, random_state = 0)

# Feature Scaling (Important for high intensity computations)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train1 = sc.fit_transform(X_train1)
#X_test1 = sc.transform(X_test1)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Fitting XGBoost to the Training set
classifier1 = XGBClassifier()
classifier1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred_X = classifier.predict(X_train)
y_pred1_X1 = classifier.predict(X_train1)
y_pred_X_test = classifier.predict(X_test)

mae_X_test = np.round(abs(y_test.values - y_pred_X_test).mean(),decimals=2)
rmse_X_test = (((y_test.values - y_pred_X_test)**2).mean())**(1/2)
r_X_test = (((y_test.values-y_test.values.mean())*(y_pred_X_test - y_pred_X_test.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred_X_test.std()))

y_test_total = ((pd.DataFrame(y_pred_test_rd) + pd.DataFrame(y_pred_X_test.ravel()))/2).astype(int)
mae_test_total = np.round(abs(y_test.values - y_test_total[0]).mean(),decimals=2)
rmse_test_total = (((y_test.values - y_test_total[0])**2).mean())**(1/2)
r_test_total = (((y_test.values-y_test.values.mean())*(y_test_total[0] - y_test_total[0].mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_test_total[0].std()))


y_pred_error = y_train.values - pd.DataFrame(y_pred_X.ravel())

# Fitting XGBoost to the Training set
classifier1 = XGBClassifier()
classifier1.fit(X_train, y_pred_error)

y_pred_Xr = classifier1.predict(X_train)
y_pred_error_r = y_pred_error - pd.DataFrame(y_pred_Xr.ravel())
y_predicted = y_pred_X + y_pred_Xr

mae_Xr = np.round(abs(y_test.values - y_predicted).mean(),decimals=2)
mae_pred = abs(y_test.values - y_pred_test_rd).mean()

rmse_Xr = (((y_test.values - y_predicted)**2).mean())**(1/2)
rmse_pred = (((y_test.values - y_pred_test_rd)**2).mean())**(1/2)

r_Xr = (((y_test.values-y_test.values.mean())*(y_pred_X - y_predicted.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_predicted.std()))
r_pred = (((y_test.values-y_test.values.mean())*(y_pred_test_rd - y_pred_test_rd.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred_test_rd.std()))

y_mean = y.mean().astype(int)
y_res = y-y_mean










# Predicting the Test set results
#y_pred1 = classifier1.predict(X_test1)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean_acc = accuracies.mean()
train_std = accuracies.std()

# Applying k-Fold Cross Validation
accuracies1 = cross_val_score(estimator = classifier1, X = X_train1, y = y_train1, cv = 10)
mean_acc1 = accuracies1.mean()
train1_std = accuracies1.std()
'''
# Applying k-Fold Cross Validation
accuracies2 = cross_val_score(estimator = classifier2, X = X_test, y = y_test, cv = 10)
mean_acc1 = accuracies1.mean()
accuracies1.mean()
accuracies1.std()'''

feature_importance = classifier.feature_importances_
features = pd.Series(feature_importance, index = X3.columns)
features_sorted = pd.Series(features.sort_values())

feature_importance1 = classifier1.feature_importances_
features1 = pd.Series(feature_importance1, index = X3.columns)
features_sorted1 = pd.Series(features1.sort_values())

'''
features.sort_values()
features_index_sort = features.index()

features1.sort_values()
features_index_sort1 = features1.index()
'''

X_train_weighted = X_train*feature_importance
X_train1_weighted = X_train1*feature_importance1
X_train_weighted_total = np.concatenate([X_train_weighted, X_train1_weighted], axis=0)
X_train_weighted_total_norm = (X_train_weighted_total - X_train_weighted_total.mean()) / (X_train_weighted_total.max() - X_train_weighted_total.min())
y_train_weighted = np.concatenate([y_train, y_train1], axis=0)
#X_test_total = np.concatenate([X_test, X_test1], axis=0)
#y_test_total = pd.DataFrame(np.concatenate([y_test, y_test1], axis=0), index = list(np.concatenate([y_test.index,y_test1.index], axis=0)))
'''
features_avg = (features + features1)/2
X_test_weighted = X_test*features_avg.values
X_test_weighted_norm = (X_test_weighted - X_test_weighted.mean()) / (X_test_weighted.max() - X_test_weighted.min())
'''

X4 = X_train_weighted_total_norm
y1 = y_train_weighted

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras import objectives
from keras import backend as K

# Initialising the ANN
regressor1 = Sequential()

# Adding the input layer and the first hidden layer with dropout (start 0.1 (10%))
regressor1.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu', input_dim = len(X3.columns)))
regressor1.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor1.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
regressor1.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor1.add(Dense(output_dim = 35, init = 'uniform', activation = 'relu'))
regressor1.add(Dropout(p=0.1))

# Adding the third hidden layer
regressor1.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
regressor1.add(Dropout(p=0.1))

# Adding the output layer
regressor1.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN (adam[SGD] - optimizer function to find optimal weights)
# Binary Dept Var(Binary_CrossEntropy) Dependent Var > 2 Outcomes (Categorical_CrossEntropy)
# [Accuracy] in brackets because list expected
regressor1.compile(optimizer = 'adam', loss = 'mse', metrics=['binary_crossentropy','acc'])

# Fitting the ANN to the Training set
regressor1.fit(X_train, y_train, batch_size = 35, epochs = 500)
'''
# Predicting the Test set results
y_pred_test = regressor1.predict(X_test)
y_pred_test_rd = np.round(y_pred_test)
y_test_index = [x for x in y_test.index]

# Predicting a new result
y_pred_train = regressor1.predict(X4)
y_pred_train_rd = np.round(y_pred_train)
y_train_index = [x for x in y.index]'''

features_avg = (features + features1)/2
X_test_weighted = X_test*features_avg.values
X_test_weighted_norm = (X_test_weighted - X_test_weighted.mean()) / (X_test_weighted.max() - X_test_weighted.min())

# Predicting the Test set results
y_pred_test = regressor1.predict(X_test_weighted_norm)
y_pred_test_rd = np.round(y_pred_test)
y_test_total_index = [x for x in pd.Series(y_test.index)]
y_pred_test1 = regressor1.predict(X_test)
y_pred_test1_rd = np.round(y_pred_test1)


expected = X2['Expected_Runs']
y_expected_test = expected[y_test_total_index]
y_expected_test_round = np.round(y_expected_test)

mae = np.round(abs(y_test.values - y_expected_test_round.values).mean(),decimals=2)
mae_pred = abs(y_test.values - y_pred_test_rd).mean()
mae_test = np.round(abs(y_test.values - y_pred_test1_rd).mean(),decimals=2)

rmse = (((y_test.values - y_expected_test_round.values)**2).mean())**(1/2)
rmse_pred = (((y_test.values - y_pred_test_rd)**2).mean())**(1/2)
rmse_test = (((y_test.values - y_pred_test1_rd)**2).mean())**(1/2)

r = (((y_test.values-y_test.values.mean())*(y_expected_test_round.values - y_expected_test_round.values.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_expected_test_round.values.std()))
r_pred = (((y_test.values-y_test.values.mean())*(y_pred_test_rd - y_pred_test_rd.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred_test_rd.std()))
r_test = (((y_test.values-y_test.values.mean())*(y_pred_test1_rd - y_pred_test1_rd.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred_test1_rd.std()))

y_test_total = ((pd.DataFrame(y_pred_test1_rd) + pd.DataFrame(y_pred_X_test.ravel()))/2).astype(int)
mae_test_total = np.round(abs(y_test.values - y_test_total).mean(),decimals=2)
rmse_test_total = (((y_test.values - y_test_total)**2).mean())**(1/2)
r_test_total = (((y_test.values-y_test.values.mean())*(y_test_total - y_test_total.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_test_total.std()))


import thinkstats2
from thinkstats2 import *
from code import *
import thinkplot

y2 = y1.flatten()

pmf_scores = thinkstats2.Pmf(y2)
thinkplot.Hist(pmf_scores)
thinkplot.Config(xlabel='Runs Scored', ylabel='probability', axis=[0, 20, 0, 0.3])

cdf_scores = thinkstats2.Cdf(y2, label='Runs Scored')
cdf_ld = thinkstats2.Cdf(X3['bat_LD%'], label='Line Drives')
cdf_pop = thinkstats2.Cdf(X3['bat_POP%'], label='Pop Ups')
cdf_gb = thinkstats2.Cdf(X3['bat_GB%'], label='Ground Balls')

thinkplot.PrePlot(4)
thinkplot.Cdfs([cdf_scores, cdf_ld, cdf_pop, cdf_gb])
thinkplot.Show(xlabel='balls in play (%)', ylabel='CDF')

# Visualizing data in One Dimension (1-D)
import matplotlib.pyplot as plt
y.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 15, 0, 15)) 

# visualizing one of the continuous, numeric attributes
# Histogram
fig = plt.figure(figsize = (10,4))
title = fig.suptitle("Runs", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.1)

ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("SDTHB_BAT")
ax.set_ylabel("Frequency") 
ax.set_xlim([0.35, 0.55])
#ax.text(0.8, 300, '%='+str(round(census_data['IncomePerCap'].dropna(),1)), fontsize=10)
freq, bins, patches = ax.hist(X3['SDTHB_BAT'].dropna(), color='steelblue', bins=15,
                                    edgecolor='black', linewidth=1)


# Density Plot
import seaborn as sns
fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("SDTHB_BAT", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,1, 1)
ax1.set_xlabel("SDTHB_BAT")
ax1.set_ylabel("Frequency") 
sns.kdeplot(X3['SDTHB_BAT'].dropna(), ax=ax1, shade=True, color='steelblue')

# Multivariate Analysis
# Visualizing data in Two Dimensions (2-D)
# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = X3.corr()
hm = sns.heatmap(round(corr,2), annot=False, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Game Correlation Heatmap', fontsize=14)


# Pair-wise Scatter Plots
cols = list(X3.columns)
pp = sns.pairplot(X3[cols[-15:]], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Game Attributes Pairwise Plots', fontsize=14)

#  parallel coordinates
# Scaling attribute values to avoid few outiers
cols = list(X3.columns)
subset_df = X3[cols[-15:]]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols[-15:])
final_df = pd.concat([scaled_df, X3['SDTHB_BAT']], axis=1)
final_df.head()


# plot parallel coordinates
from pandas.plotting import parallel_coordinates
pc = parallel_coordinates(final_df, 'SDTHB_BAT', color=('#FFE888', '#FF9999', 'DarkGreen'))

# visualize two continuous, numeric attributes. Scatter plots and joint plots
# Scatter Plot
plt.scatter(X3['SDTHB_BAT'], y,
            alpha=0.4, edgecolors='w')

plt.xlabel('SDTHB_BAT')
plt.ylabel('bat_GB%')
plt.title('SDTHB_BAT - bat_GB%',y=1.05)

# Joint Plot
jp = sns.jointplot(x='SDTHB_BAT', y='bat_GB%', data=X3,
                   kind='reg', space=0, size=5, ratio=4)







# visualizing two discrete, categorical attributes
# Using subplots or facets along with Bar Plots
fig = plt.figure(figsize = (10, 4))
title = fig.suptitle("TB_batter - Era", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
# red wine - wine quality
ax1 = fig.add_subplot(1,2, 1)
ax1.set_title("TB_batter for pitchers with ERA <= 3.5")
ax1.set_xlabel("TB_batter")
ax1.set_ylabel("Era") 
rw_q = X3['TB_batter'][X3['pitcher_ERA']<=3.5].value_counts()
rw_q = (list(rw_q.index), list(rw_q.values))
ax1.set_ylim([0, 2500])
ax1.tick_params(axis='both', which='major', labelsize=8.5)
bar1 = ax1.bar(rw_q[0], rw_q[1], color='red', 
               edgecolor='black', linewidth=1)

# white wine - wine quality
ax2 = fig.add_subplot(1,2, 2)
ax2.set_title("New York")
ax2.set_xlabel("Unemployment")
ax2.set_ylabel("Frequency") 
ww_q = census_drop['Unemployment_Rate'][census_drop['State']=='New York'].value_counts()
ww_q = (list(ww_q.index), list(ww_q.values))
ax2.set_ylim([0, 2500])
ax2.tick_params(axis='both', which='major', labelsize=8.5)
bar2 = ax2.bar(ww_q[0], ww_q[1], color='white', 
               edgecolor='black', linewidth=1)


xint = census_drop["Unemployment"].astype(int)
# stacked bars or multiple bars
# Multi-bar Plot
cp = sns.countplot(x=xint, hue="Unemployment_Rate", data=census_drop, 
                   palette={"high": "#FF9999", "medium": "#FFE888", "low": '#001C7F'})


# visualizing mixed attributes in two-dimensions
#  faceting\subplots along with generic histograms or density plots.
# facets with histograms
fig = plt.figure(figsize = (10,4))
title = fig.suptitle("pitcher_ERA vs runners that touched base", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,3, 1)
ax1.set_title("Low TB's")
ax1.set_xlabel("pitcher_ERA")
ax1.set_ylabel("Frequency") 
ax1.set_ylim([0, 200])
ax1.text(1.2, 800, r'$mu$='+str(round(X3['pitcher_ERA'][X3['TB_batter']<25].mean(),2)), 
         fontsize=12)
r_freq, r_bins, r_patches = ax1.hist(X3['pitcher_ERA'][X3['TB_batter']<25], color='red', bins=15,
                                     edgecolor='Black', linewidth=1)

ax2 = fig.add_subplot(1,3, 2)
ax2.set_title("Medium TB's")
ax2.set_xlabel("pitcher_ERA")
ax2.set_ylabel("Frequency")
ax2.set_ylim([0, 200])
ax2.text(0.8, 800, r'$mu$='+str(round(X3['pitcher_ERA'][X3['TB_batter']<45].mean(),2)), 
         fontsize=12)
w_freq, w_bins, w_patches = ax2.hist(X3['pitcher_ERA'][X3['TB_batter']<45], color='white', bins=15,
                                     edgecolor='Black', linewidth=1)


ax3 = fig.add_subplot(1,3, 3)
ax3.set_title("High TB's")
ax3.set_xlabel("pitcher_ERA")
ax3.set_ylabel("Frequency")
ax3.set_ylim([0, 200])
ax3.text(0.8, 800, r'$mu$='+str(round(X3['pitcher_ERA'][X3['TB_batter']>=45].mean(),2)), 
         fontsize=12)
w_freq, w_bins, w_patches = ax3.hist(X3['pitcher_ERA'][X3['TB_batter']>=45], color='green', bins=15,
                                     edgecolor='Black', linewidth=1)


# facets with density plots
fig = plt.figure(figsize = (10, 4))
title = fig.suptitle("Unemployment Content in USA", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.5)

ax1 = fig.add_subplot(1,3, 1)
ax1.set_title("Low Unemployment")
ax1.set_xlabel("Black")
ax1.set_ylabel("Density") 
sns.kdeplot(X3['Black'][X3['Unemployment_Rate']=='low'], ax=ax1, shade=True, color='r')

ax2 = fig.add_subplot(1,3, 2)
ax2.set_title("Medium Unemployment")
ax2.set_xlabel("Black")
ax2.set_ylabel("Density") 
sns.kdeplot(X3['Black'][X3['Unemployment_Rate']=='medium'], ax=ax2, shade=True, color='g')

ax3 = fig.add_subplot(1,3, 3)
ax3.set_title("High Unemployment")
ax3.set_xlabel("Black")
ax3.set_ylabel("Density") 
sns.kdeplot(X3['Black'][X3['Unemployment_Rate']=='high'], ax=ax3, shade=True, color='y')


# Using multiple Histograms 
fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Unemployment Content in USA", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Black")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(X3, hue='Unemployment_Rate', palette={"high": "r", "medium": "y", 'low': 'g'})
g.map(sns.distplot, 'Black', kde=False, bins=15, ax=ax)
ax.legend(title='Unemployment')
plt.close(3)


# Box Plots
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Unemployment - Black', fontsize=14)

sns.boxplot(x="Unemployment_Rate", y="Black", data=X3,  ax=ax)
ax.set_xlabel("Unemployment",size = 12,alpha=0.8)
ax.set_ylabel("Black %",size = 12,alpha=0.8)


# Violin Plots
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Unemployment - Black', fontsize=14)

sns.violinplot(x="Unemployment_Rate", y="Black", data=X3,  ax=ax)
ax.set_xlabel("Unemployment_Rate",size = 12,alpha=0.8)
ax.set_ylabel("Black",size = 12,alpha=0.8)


# Visualizing data in Three Dimensions (3-D)
# pair-wise scatter plot 
# Scatter Plot with Hue for visualizing data in 3-D
cols = ['CensusTract', 'TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 'Citizen',
       'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment', 'Unempoyment_Rate']
pp = sns.pairplot(X3[cols], hue='Unemployment_Rate', size=1.8, aspect=1.8, 
                  palette={"high": "#FF9999", "medium": "#FFE888", "low": '#001C7F'},
                  plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Unemployment_Rate Pairwise Plots', fontsize=14)


# visualizing three continuous, numeric attributes
# Visualizing 3-D numeric data with Scatter Plots
# length, breadth and depth
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = X3['Black']
ys = X3['Professional']
zs = X3['Unemployment']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('Black')
ax.set_ylabel('Professional')
ax.set_zlabel('Unemployment Rate')


# # Visualizing 3-D numeric data with a bubble chart
# length, breadth and size
plt.scatter(X3['Black'], X3['Professional'], s=X3['Unemployment']*25, 
            alpha=0.4, edgecolors='w')

plt.xlabel('Professional')
plt.ylabel('Black')
plt.title('Black - Professional - Unemployment',y=1.05)


# visualizing three discrete, categorical attributes
# Visualizing 3-D categorical data using bar plots
# leveraging the concepts of hue and facets
fc = sns.factorplot(x="Black", hue="Unemployment_Rate", col="Unemployment_Rate", 
                    data=X3, kind="count",
                    palette={"high": "#FF9999", "medium": "#FFE888", "low": '#001C7F'})



# three mixed attributes
# Visualizing 3-D mix data using scatter plots
# leveraging the concepts of hue for categorical dimension
jp = sns.pairplot(X3, x_vars=["Black"], y_vars=["Professional"], size=4.5,
                  hue="Unemployment_Rate", palette={"high": "#FF9999", "medium": "#FFE888", "low": '#001C7F'},
                  plot_kws=dict(edgecolor="k", linewidth=0.5))
                  
# we can also view relationshipscorrelations as needed                  
lp = sns.lmplot(x='Black', y='Professional', hue='Unemployment_Rate', 
                palette={"high": "#FF9999", "medium": "#FFE888", "low": '#001C7F'},
                data=X3, fit_reg=True, legend=True,
                scatter_kws=dict(edgecolor="k", linewidth=0.5))    


    
# Visualizing 3-D mix data using kernel density plots
# leveraging the concepts of hue for categorical dimension
ax = sns.kdeplot(X3['Black'][X3['Unemployment_Rate']=='high'], X3['Professional'][X3['Unemployment_Rate']=='high'],
                  cmap="YlOrBr", shade=True, shade_lowest=False)
ax = sns.kdeplot(X3['Black'][X3['Unemployment_Rate']=='low'], X3['Professional'][X3['Unemployment_Rate']=='low'],
                  cmap="Reds", shade=True, shade_lowest=False)
    
    

# Predicting the Full Results
y_pred_full = regressor.predict(X3)
y_pred_full_rd = np.round(y_pred_full)

# Predicting a new result
y_pred_train = regressor.predict(X_train)
y_pred_train_rd = np.round(y_pred_train)
y_score_train = regressor.evaluate(X_train, y_train, batch_size = 10)
y_score_test = regressor.evaluate(X_test, y_test, batch_size = 10)




feature_importance = classifier.feature_importances_
features = pd.Series(feature_importance, index=list(X.columns))

features.sort_values()
features_index_sort = features.sort_values()
features_index_sort1 = features_index_sort.ravel()

features_top = [x for x in features_index_sort1 if x>0.005]
features_top_name = features_index_sort[-65:]
features_top_list = [str(x) for x in features_top_name.index]





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:31:56 2018

@author: ashish
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

train=pd.read_csv('../houseprices/train.csv')
test=pd.read_csv('../houseprices/test.csv')

##difference between train and test 
train.columns.difference(test.columns)

##COMBINE train and test data .Separate labels from train data

target='SalePrice'
labels=train[target]
train['Test']=0
test['Test']=1
full=pd.concat([train.drop(target,axis=1),test])


##LOOK at data
full.info()
full.describe()

full.hist(bins=50,figsize=(30,20))
plt.show()

#DATA EXploration
data_explore=train.copy()

#correlations
correl_matrix=data_explore.corr()
correl_matrix=correl_matrix[target].sort_values(ascending=False)
correl_matrix

high_correl=[]
for i in correl_matrix.index:
    if abs(correl_matrix[i])>=0.5:
        high_correl.append(i)
        
high_correl

scatter_matrix(data_explore[high_correl],figsize=(20,15))
plt.show()

linear_corr=['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF']
for i in linear_corr:
    data_explore.plot(kind='scatter',x=target,y=i)
    
#Preprocessing data
#filling missing values case by case describe(),alue_counts()
full.info()

#applying imputer to mostattributes
median_imputer=SimpleImputer(strategy='median')
freq_imputer=SimpleImputer(strategy='most_frequent')
median_fix= ['LotFrontage']
freq_fix = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
    'KitchenQual', 'Functional', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
    'GarageArea', 'GarageQual', 'GarageCond', 'SaleType']
drop_columns = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

def impute(data,predictor,imputer):
    for i in predictor:
        data[i]=imputer.fit_transform(data[[i]]).ravel()
        
#missing values filled
impute(full,median_fix,median_imputer)
impute(full,freq_fix,freq_imputer)
full=full.drop(drop_columns,axis=1)
full.info()
        
#one hot encoding categorical variables

def onehot_encode(data):
    predictor=data.select_dtypes(include=[object]).columns.tolist()
    encoder=OneHotEncoder(sparse=False)
    for i in predictor:
        data[i]=encoder.fit_transform(data[[i]])
        
full_coded=full.copy()
onehot_encode(full_coded)

#SCALING

#separate data n scale separately
train_coded=full_coded.loc[full_coded['Test']==0].drop('Test',axis=1)
test_coded=full_coded.loc[full_coded['Test']==1  ].drop('Test',axis=1)   

def std_scaled(data,scaler,is_train):
    preds=data.select_dtypes(include=[np.int,np.float]).columns.tolist()[1:]
    if is_train:
        for i in preds:
            data[i]=scaler.fit_transform(data[[i]])
    else:
        for i in preds:
            data[i]=scaler.transform(data[[i]])
    return data
        
##CAPTURE data

scaler=StandardScaler()
train_prepared=std_scaled(train_coded,scaler,True)   
test_prepared=std_scaled(test_coded,scaler,False)

#MOdelling
    
forest_reg=RandomForestRegressor()
##capture
forest_reg.fit(train_prepared,labels)               

test_data=train_prepared[:5]
actual_value=labels[:5]
print("forest regression predictions",forest_reg.predict(test_data))
print("actual labels",list(actual_value))
forest_reg_predicted=forest_reg.predict(train_prepared)
forest_reg_rmse=np.sqrt(mean_squared_error(labels,forest_reg_predicted))
forest_reg_rmse
    
    ##orest regression predictions [206250. 172450. 222390. 146430. 250000.]
############################actual labels [208500, 181500, 223500, 140000, 250000]
    ##tuning
    
        
##capture

parameters={'n_estimators':randint(low=1,high=200),
            'max_features':randint(low=1,high=80)}

forest_reg = RandomForestRegressor(random_state=7)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=parameters,                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=7)
rnd_search.fit(train_prepared, labels)
final_model = rnd_search.best_estimator_
final_model_predicted = final_model.predict(train_prepared)
final_model_rmse = np.sqrt(mean_squared_error(labels, final_model_predicted))
print("final_model_rmse",final_model_rmse)

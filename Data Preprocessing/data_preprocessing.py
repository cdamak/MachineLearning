# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:58:03 2018

@author: karan_dama
"""

##Data preprocessing template

## Importing Libraries

import numpy as np          
import matplotlib.pyplot as plt 
import pandas as pd

## Importing dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Replace Missing Data with Mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean' , axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] =imputer.transform(x[:,1:3])

#Encode Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le_country = LabelEncoder()
x[:,0] = le_country.fit_transform(x[:,0])
ohe_country = OneHotEncoder(categorical_features = [0])
x = ohe_country.fit_transform(x).toarray()
le_purchase = LabelEncoder()
y = le_purchase.fit_transform(y)

#Split data Set into Training Set and Test Set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


#feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x  = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)




















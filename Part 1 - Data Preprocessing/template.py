#Data Preprocessing
##importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# -*- coding: utf-8 -*-
#importing the dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:,3].values

np.set_printoptions(threshold=np.nan)
#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=0) #genelde 0.2 0.25 civarı

"""#feature scaling
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_texr =sc_X.transform(X_test)"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import losses
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame



#Bring in all data for all years. This notebook contains Joseph's classifiers from 0-3 for risk groups. Col 151 is risk groups.
ct_sheet = pd.ExcelFile("ctAutoEncoder.xlsx") 


# In[2]:


#Will create a large function to apply to the 4 CT Scans for each year.
def prediction(spreadsheet):
    parsee = spreadsheet.sheet_names[0]
    data = spreadsheet.parse(parsee)
    data_features = data.loc[:, data.columns] 
    data_features = data_features.drop(['norm_id','AtRisk',11142,12142,'Age'], axis=1) 
    #Get rid of subject names to only have features now. #Need to remove ROIs. They don't convert to floats.
    #Get rid of ctx_rh_Medial_wall and ctx_lh_Medial_wall, not needed for analysis.
    #Have to standardize data. Scikit learn here. Need to create stratified K folds to avoid uneven distribution of risk groups.pcaCT1Y = PCA(n_components=150) #150 Features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_features)
    print(scaled_data.shape)
    X_train, X_test = train_test_split(scaled_data, test_size=0.10, random_state=20)
    #X = scaled_data
    #y = data['AtRisk']
    #skf = StratifiedKFold(n_splits=10, random_state=55)
    #print(skf.get_n_splits(data_features, y))
    #for train_index, test_index in skf.split(X, y):
        #X_train, X_test = X[train_index], X[test_index] 
        #print(split)
        #X_train = split[0][0] 
        #Gives me row numbers such that I can keep stratified folds. But how do I input the data from these rows?
        #X_test = split[0][1]
        #print(X_test)
        #print(X_train)
        #print(y)
        #print(skf) self, X, y, groups=None
        #dataFeatures.head()
        #X_train, X_test = train_test_split(scaled_data, test_size=0.10, random_state=20)

        #Size of encoded representation.
    input_size = 148
    hidden_size = 74
    encoding_dim = 13 # 13 floats -> compression of factor ~11.5, assuming the input is 150 floats

    # Input Placeholder
    input_data = Input(shape=(input_size,))
    print(input_data)
    # "encoded" is the encoded representation of the input
    hidden_e_1 = Dense(hidden_size, activation='tanh')(input_data) 
    encoded = Dense(encoding_dim, activation='tanh')(hidden_e_1)
    # "decoded" is the lossy reconstruction of the input
    hidden_d_1 = Dense(hidden_size, activation='tanh')(encoded)
    decoded = Dense(input_size, activation='tanh')(hidden_d_1) #Decoded layers and activation function. Needs to return to 151.
    # this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded)
    # configure our model to use mean_squared_error loss function, and the Adadelta optimizer:
    autoencoder.compile(optimizer='Adam', loss='mean_absolute_error')

    ac = autoencoder.fit(X_train, X_train,
    epochs=5000,
    batch_size=15,
    shuffle=True,
    validation_data=(X_test, X_test))

    #print(ac.history.keys())
    # "Loss"
    plt.plot(ac.history['loss'])
    plt.plot(ac.history['val_loss'])
    #plt.set(xlim=(0, 50), ylim=(0.0, 1.0))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.axis([0, 5000, 0.0, 1.0])
    plt.show()


# In[3]:


#Need to put my training and testing data here.
#I have the features scaled. Now how do I:
# 1) Split them into balanced folds using stratified kfolds?
# 2) Input them into the appropriate training/testing slots?


# In[4]:


prediction(ct_sheet)


# In[5]:


get_ipython().run_line_magic('pinfo', 'skf')


# In[ ]:





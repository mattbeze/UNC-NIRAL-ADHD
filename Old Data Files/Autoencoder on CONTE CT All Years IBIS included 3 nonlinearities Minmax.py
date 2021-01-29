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
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import xlsxwriter



#Bring in all data for all years. This notebook contains Joseph's classifiers from 0-3 for risk groups. Col 151 is risk groups.
ct_sheet = pd.ExcelFile("SA_and_CT_AALandfROI_08272019.xlsx") 


# In[2]:


print(ct_sheet.sheet_names[9])


# In[3]:


#Will create a large function to apply to the 4 CT Scans for each year.
parsee = ct_sheet.sheet_names[9]
data = ct_sheet.parse(parsee)
data_features = data.loc[:, data.columns] 
data_features = data_features.drop(['Case','Visit','ROI42','ROI117'], axis=1)  
#Get rid of subject names to only have features now. #Need to remove ROIs. They don't convert to floats.
#Get rid of ctx_rh_Medial_wall and ctx_lh_Medial_wall, not needed for analysis.
#Have to standardize data. Scikit learn here. Need to create stratified K folds to avoid uneven distribution of risk groups.pcaCT1Y = PCA(n_components=150) #150 Features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_features)

ct_sheet = pd.ExcelFile("ctAutoEncoder.xlsx") 
parsee = ct_sheet.sheet_names[0]
data = ct_sheet.parse(parsee)
data_features = data.loc[:, data.columns] 
data_features = data_features.drop(['norm_id','AtRisk',11142,12142,'Age'], axis=1)  
scaled_data = scaler.transform(data_features)

print(scaled_data.shape)
X_train, X_test = train_test_split(scaled_data, test_size=0.10, random_state=20)

#Size of encoded representation
input_size = 148
hidden_size3 = 120
hidden_size2 = 65
hidden_size1 = 30
encoding_dim = 13 # 13 floats -> compression of factor ~11.5, assuming the input is 150 floats

# Input Placeholder
input_data = Input(shape=(input_size,))
print(input_data)
# "encoded" is the encoded representation of the input
hidden_e_3 = Dense(hidden_size3, activation='tanh')(input_data)
hidden_e_2 = Dense(hidden_size2, activation='tanh')(hidden_e_3)
hidden_e_1 = Dense(hidden_size1, activation='tanh')(hidden_e_2) 
encoded = Dense(encoding_dim, activation='tanh')(hidden_e_1)
# "decoded" is the lossy reconstruction of the input
hidden_d_1 = Dense(hidden_size1, activation='tanh')(encoded)
hidden_d_2 = Dense(hidden_size2, activation='tanh')(hidden_d_1)
hidden_d_3 = Dense(hidden_size3, activation='tanh')(hidden_d_2)
decoded = Dense(input_size, activation='tanh')(hidden_d_3) #Decoded layers and activation function. Needs to return to 151.
# this model maps an input to its reconstruction
autoencoder = Model(input_data, decoded)
# configure our model to use mean_absolute_error loss function, and the Adam optimizer:
autoencoder.compile(optimizer='Adam', loss='mean_absolute_error')

ac = autoencoder.fit(X_train, X_train,
epochs=1500,
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
plt.axis([0, 1500, 0.0, 0.15])
plt.show()


# In[4]:


sa_data = pd.ExcelFile("CTGilmore.xlsx") 
parsee = sa_data.sheet_names[0]
data = sa_data.parse(parsee)
data_features = data.loc[:, data.columns] 
data_features = data_features.drop(['norm_id','AtRisk','Age',11142,12142], axis=1)  
scaled_data = scaler.transform(data_features)

encoder = Model(input_data, encoded)
encoded_data = encoder.predict(scaled_data)
#I now have scaled data as encoded which is my input. This will compare predictions to my scaled_data.
Y_train, Y_test = train_test_split(scaled_data, test_size=0.10, random_state=20) #Scaled data I want to compare to
X_train, X_test = train_test_split(encoded_data, test_size=0.10, random_state=20) #Encoded data I will input

dinput_size = 13
hidden_size1 = 30
hidden_size2 = 65
hidden_size3 = 120
decoded_dim = 148
dinput_data = Input(shape=(dinput_size,))
dhidden_d_1 = Dense(hidden_size1, activation='tanh')(dinput_data)
dhidden_d_2 = Dense(hidden_size2, activation='tanh')(dhidden_d_1)
dhidden_d_3 = Dense(hidden_size3, activation='tanh')(dhidden_d_2)
predictedSA = Dense(decoded_dim, activation='tanh')(dhidden_d_3)

predictor = Model(dinput_data, predictedSA)
predictor.compile(optimizer='Adam', loss='mean_absolute_error')
pn = predictor.fit(X_train, Y_train,
epochs=1500,
batch_size=15,
shuffle=True,
validation_data=(X_test, Y_test))

#print(ac.history.keys())
# "Loss"
plt.plot(pn.history['loss'])
plt.plot(pn.history['val_loss'])
#plt.set(xlim=(0, 50), ylim=(0.0, 1.0))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.axis([0, 1500, 0.0, 0.15])
plt.show()


# In[ ]:





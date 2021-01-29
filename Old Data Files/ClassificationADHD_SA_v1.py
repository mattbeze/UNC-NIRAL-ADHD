#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This notebook is an illustrative example for using the SA Conte data for classification.
While it loads also the twin data, that data is not used. The twin data should only be used once we are all done 
with trying things out and want to see whether the signal that we have found is real
The conte data is used in a 3-fold analysis, each fold is upsampled for balance via smote 
3 classifications are applied. SVM (rbf kernel), random forrest and a fully connected DL network

The SVM results are the leasy overfit, though also the poorest in performance. The random forrest results are overfit, 
such that it always returns a NR class. The DL results are best, though also clearly biased towards the NR class. 
Changing around with network parameters does not really seem to improve the situation. 

'''
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import numpy as np
from numpy import argmax
from numpy.random import seed
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import optimizers
from tensorflow import set_random_seed


# In[ ]:


sa1y = pd.read_csv('sa1y_basc_brief.csv').set_index('norm_id')
sa1y = sa1y[sa1y['AtRisk'] != 0]


# In[ ]:


twins = sa1y[sa1y['study'] == 'TWIN']
X_twin = twins.loc[:, ~sa1y.columns.isin(['AtRisk', 'ROI', 'study'])].values
y_twin= twins['AtRisk'].values
Counter(y_twin)


# In[ ]:


contes = sa1y[sa1y['study'] == 'CONTE']
X_conte = contes.loc[:, ~sa1y.columns.isin(['norm_id','AtRisk', 'ROI', 'study'])].values
y_conte = contes['AtRisk'].values
Counter(y_conte)


# In[ ]:


# Norm with 10% enlarged min and max from twins
MinVals = np.array(np.amin(X_twin, axis = 0)) * 0.8
MaxVals = np.array(np.amax(X_twin, axis = 0)) * 1.2
NormScale = (MaxVals - MinVals)
NormSub = MinVals
X_twin_norm = (X_twin - NormSub) / NormScale
X_conte_norm = (X_conte - NormSub) / NormScale


# In[ ]:


reducer = umap.UMAP()
embedding = reducer.fit_transform(X_conte_norm)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in y_conte])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of Surface Area at 1y and ADHD Risk for Conte', fontsize=24);


# In[ ]:


reducer = umap.UMAP()
embedding = reducer.fit_transform(X_twin_norm)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in y_twin])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of Surface Area at 1y and ADHD Risk for Conte', fontsize=24);


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
y_allPred = []
y_allVal = []
foldNumber = 0
for train_index, val_index in skfold.split(X_conte_norm,y_conte):
    X_foldtrain, X_foldval = X_conte_norm[train_index], X_conte_norm[val_index]
    y_foldtrain, y_foldval = y_conte[train_index], y_conte[val_index]
    
    print(foldNumber, Counter(y_foldval))
    foldNumber+=1
    
    #Smote will oversample the minority classes
    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_sample(X_foldtrain, y_foldtrain)
    
    #first SVM
    clf = svm.SVC(gamma='auto', kernel='rbf', decision_function_shape='ovo')
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_foldval)
    y_allPred = np.append(y_allPred, y_pred)
    y_allVal = np.append(y_allVal, y_foldval)
    print (accuracy_score(y_foldval, y_pred))

print("All folds")
print (accuracy_score(y_allVal, y_allPred))
plot_confusion_matrix(confusion_matrix(y_allVal, y_allPred), classes=['LR','MR','HR'])


# In[ ]:


skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
y_allPred = []
y_allVal = []
foldNumber = 0
for train_index, val_index in skfold.split(X_conte_norm,y_conte):
    X_foldtrain, X_foldval = X_conte_norm[train_index], X_conte_norm[val_index]
    y_foldtrain, y_foldval = y_conte[train_index], y_conte[val_index]
    
    print(foldNumber, Counter(y_foldval))
    foldNumber+=1
    
    #Smote will oversample the minority classes
    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_sample(X_foldtrain, y_foldtrain)
    
    #Random Forest
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_res,y_train_res)
    y_pred = clf.predict(X_foldval)
    y_allPred = np.append(y_allPred, y_pred)
    y_allVal = np.append(y_allVal, y_foldval)
    print (accuracy_score(y_foldval, y_pred))

print("All folds")
print (accuracy_score(y_allVal, y_allPred))
plot_confusion_matrix(confusion_matrix(y_allVal, y_allPred), classes=['LR','MR','HR'])


# In[ ]:


skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
y_allPred = []
y_allVal = []
foldNumber = 0
for train_index, val_index in skfold.split(X_conte_norm,y_conte):
    X_foldtrain, X_foldval = X_conte_norm[train_index], X_conte_norm[val_index]
    y_foldtrain, y_foldval = y_conte[train_index], y_conte[val_index]
    
    print(foldNumber, Counter(y_foldval))
    foldNumber+=1
    
    #Smote will oversample the minority classes
    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_sample(X_foldtrain, y_foldtrain)
    
    #DL
    y_train_labels = to_categorical(y_train_res- 1,num_classes=3)
    y_val_labels = to_categorical(y_foldval - 1,num_classes=3)
    
    network = models.Sequential()
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(5, activation='relu'))
    network.add(layers.Dense(3,activation='softmax'))
    network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    fitProcess = network.fit(X_train_res, y_train_labels, epochs=200, batch_size = 40, 
                             verbose = 0, validation_data=(X_foldval, y_val_labels))
    y_pred=argmax(network.predict(X_foldval),axis = 1) + 1
    
    y_allPred = np.append(y_allPred, y_pred)
    y_allVal = np.append(y_allVal, y_foldval)
    print (accuracy_score(y_foldval, y_pred))

print("All folds")
print (accuracy_score(y_allVal, y_allPred))
plot_confusion_matrix(confusion_matrix(y_allVal, y_allPred), classes=['LR','MR','HR'])


# In[ ]:


history_dict = fitProcess.history
history_dict.keys()


# In[ ]:


plt.clf()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.legend()
plt.show()


# In[ ]:


plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1,len(acc_values) + 1)
plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.legend()
plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import math
import preprocessing

import numpy
import pandas
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Custom imports
from utils.dataset_loader import ParkinsonDataset

def mlp():
   
    X_train, X_test, y_train, y_test = preprocessing.load_dataset('./parkinsons_updrs.csv')
                                                                  

    
    #X_all = df[ParkinsonDataset.FEATURES].values
    #y_all = df[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

    

    hidden_units = [500, 400, 300, 200]
    activation = 'sigmoid'
    lr = 0.0005

    results = pandas.DataFrame(columns=['Total-Test', 'Motor-Test'],
                               index=["All"])

    # ALL
    total_results, motor_results = [], []
    K = 5
    for i in range(K):
        
        model = keras.Sequential()
        for layer in range(len(hidden_units)):
            model.add(layers.Dense(units=hidden_units[layer], activation=activation))
        # 2 units in the output layer (Total and Motor)
        model.add(layers.Dense(units=2))
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=1000,
                            validation_split=0.1,
                            verbose=0,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

        y_pred = model.predict(X_test)

        mae_total = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_motor = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["All", "Total-Test"] = total_results
    results.at["All", "Motor-Test"] = motor_results
    print(results)


# In[ ]:





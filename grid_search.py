# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:27:46 2023

@author: he_98
"""

# Data preprocessing

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

df = pd.read_csv("bucks_roster_stats2.csv")
df = df.drop(columns = ['GAME_ID', 'MIN_TEAM', 'PF_TEAM', 'PLUS_MINUS_TEAM'])

loc = df['HOME/AWAY']
loc.replace('Home', 0, inplace=True)
loc.replace('Away', 1, inplace=True)

#df.dropna(inplace=True)
y=df['WL']
y.replace('L',0,inplace=True)
y.replace('W',1,inplace=True)
Y = tf.keras.utils.to_categorical(y)

#df.fillna(df.mean(), inplace=True)
#df.fillna(df.median(), inplace=True)
#df.fillna(df.mode(), inplace=True)
df.fillna(0, inplace=True)

#x = df.iloc[:,1:]
x = df.drop(columns = ['WL'])
x = preprocessing.scale(x)

column_names = list(df.columns)
column_names = ['HOME/AWAY'] + column_names[2:]

information = mutual_info_classif(x,y)
print('Information=',information)


feature_info = {i: info for i, info in enumerate(information)}

sorted_info = sorted(feature_info.items(), key=lambda x: x[1], reverse=True)

print("Ranked features based on mutual information scores:")
for i, info in sorted_info:
    print(f"{column_names[i]}: {info}")

top_features = np.where(information >= np.percentile(information, 75), True, False)
top_features = list(top_features)

X = x[:, top_features]
print(X.shape[1])


x_train, x_test, y_train, y_test = train_test_split (X,Y, test_size=0.2, random_state=1)

N_train = len(x_train[0])
K = Y.shape[1]

# define the parameter grid
param_grid = {
    'epochs': [25, 50, 100, 150],
    'batch_size': [25, 50, 100, 150],
    'optimizer': ['adam', 'sgd'],
    'activation': ['sigmoid', 'relu', 'tanh'],
    'neurons': [(80, 60, 40, 20), (130, 100, 50, 20), (200, 150, 100, 50)],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5]
}

# define the model function to be used by GridSearchCV
def create_model(neurons=(20, 40, 60, 80), activation='sigmoid', optimizer='adam',dropout_rate=0.5):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(neurons[0], activation=activation, input_dim=N_train),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons[1], activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons[2], activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons[3], activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(K, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

keras_model = KerasClassifier(build_fn=create_model, verbose=1)
# create the GridSearchCV object
grid_search = GridSearchCV(estimator=keras_model,
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           scoring='accuracy')

# perform the grid search
grid_search.fit(x_train, y_train)

# print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")


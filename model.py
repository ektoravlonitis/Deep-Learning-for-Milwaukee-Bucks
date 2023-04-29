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
from sklearn.decomposition import PCA
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


df.to_csv('processed_data2.csv', index=False)

#df.fillna(df.mean(), inplace=True)
#df.fillna(df.median(), inplace=True)
#df.fillna(df.mode(), inplace=True)
df.fillna(0, inplace=True)

#x = df.iloc[:,1:]
x = df.drop(columns = ['WL'])
x = preprocessing.scale(x)

#df.to_csv('processed_data2.csv', index=False)

'''
# Apply PCA
pca = PCA(n_components=80) # specify the number of components to keep
x = pca.fit_transform(x)
'''

# Performing mutual information
column_names = list(df.columns)
column_names = ['HOME/AWAY'] + column_names[2:]

information = mutual_info_classif(x,y)
print('Information=',information)


feature_info = {i: info for i, info in enumerate(information)}

sorted_info = sorted(feature_info.items(), key=lambda x: x[1], reverse=True)

print("Ranked features based on mutual information scores:")
for i, info in sorted_info:
    print(f"{column_names[i]}: {info}")

top_features = np.where(information >= np.percentile(information, 90), True, False)
top_features = list(top_features)

X = x[:, top_features]
print(X.shape[1])

# Split data to train and test
x_train, x_test, y_train, y_test = train_test_split (X,Y, test_size=0.2, random_state=1)

N_train = len(x_train[0])
K = Y.shape[1]
# Insert a dropout rate
dropout_rate = 0.2

# Use keras to create the model, with the best parameters from grid search
my_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(80, activation=tf.nn.sigmoid, input_dim=N_train),
    tf.keras.layers.Dropout(dropout_rate),  # Add dropout layer with rate of 0.2
    tf.keras.layers.Dense(60, activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(40, activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(K, activation=tf.nn.softmax)
])

my_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

my_model.fit(x_train, y_train, epochs=25, batch_size=25)
res = my_model.evaluate(x_test, y_test)
my_model.save('my_model.h5')
print('[test loss, test acc]=', res)

# Train the model
history = my_model.fit(x_train, y_train, epochs=100, batch_size=25, validation_data=(x_test, y_test))

y_pred = my_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(np.argmax(y_test,axis=1), y_pred))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

# Evaluate the model on the test data
test_loss, test_acc = my_model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

import matplotlib.pyplot as plt

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
columns_left = np.array(column_names)
print(columns_left[top_features])

for i, info in sorted_info:
    print(f"{column_names[i]}: {info}")
    '''
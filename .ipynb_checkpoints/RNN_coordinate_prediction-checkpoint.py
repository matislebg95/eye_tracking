import warnings
warnings.filterwarnings('ignore')

import numpy                   as np
import matplotlib.pyplot       as plt
import pandas                  as pd
import tensorflow 	           as tf
from   tensorflow.keras        import layers, models, Input, Sequential
from   sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

df = pd.read_csv('dataset.csv', sep = ';')

NB_OF_KEYPOINTS = 468
NB_OF_NEURALS = 64

model = tf.keras.Sequential([
    tf.keras.Input(shape = NB_OF_KEYPOINTS * 2,)),
    tf.keras.layers.Dense(NB_OF_NEURALS, activation='relu'),
    tf.keras.layers.Dense(NB_OF_NEURALS, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.summary()

def split_data(df):
    #shuffle rows
    df.sample(frac=1)

    X = np.array(df.drop(['x_coords', 'y_coords'], axis = 1))
    y = np.array(df['x_coords', 'y_coords'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return(X_train, X_test, y_train, y_test)

def train_model(model, X_train, X_test, y_train, y_test, epochs = 16):
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    history = model.fit(x = X_train,
                        y = y_train,
                        epochs = epochs,
                        validation_data = (X_test, y_test))

    return(model)


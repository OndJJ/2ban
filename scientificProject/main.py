import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

np.random.seed(3)
tf.random.set_seed(3)

df = np.loadtxt('deeplearning/run_project/01_My_First_Deeplearning.ipynb, delimiter=',')

X = df[:,0:17]
Y = df[:,17]

model = Sequential()
model.add(Dense(30, input_dim =17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accur'])
model.fit(X, Y, epochs=100, batch_size=10)

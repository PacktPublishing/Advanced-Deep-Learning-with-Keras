'''
A Simple RNN model with 30 x 12 input and 5-dim one-hot vector
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# keras modules
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam

timesteps = 30
input_dim = 12
# number of units in RNN cell
units = 512
# number of classes to be identified
n_activities = 5
model = Sequential()
# RNN with dropout
model.add(SimpleRNN(units=units,
                    dropout=0.2,
                    input_shape=(timesteps, input_dim)))
# classifier stage
model.add(Dense(n_activities, activation='softmax'))
# model loss function and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

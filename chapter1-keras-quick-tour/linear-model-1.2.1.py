'''A simple MLP in Keras implementing linear regression.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# numpy package
import numpy as np

# keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# generate x data
x = np.arange(-1,1,0.2)
x = np.reshape(x, [-1,1])

# generate y data
y = 2 * x + 3

# True if noise is added to y
is_noisy = True

# add noise if enabled
if is_noisy:
    noise = np.random.uniform(-0.1, 0.1, x.shape)
    x = x + noise

# deep learning method
# build 2-layer MLP network 
model = Sequential()
# 1st MLP has 8 units (perceptron), input is 1-dim
model.add(Dense(units=8, input_dim=1))
# 2nd MLP has 1 unit, output is 1-dim
model.add(Dense(units=1))
# print summary to double check the network
model.summary()
# create a nice image of the network model
plot_model(model, to_file='linear-model.png', show_shapes=True)
# indicate the loss function and use stochastic gradient descent
# (sgd) as optimizer
model.compile(loss='mse', optimizer='sgd')
# feed the network with complete dataset (1 epoch) 100 times
# batch size of sgd is 4
model.fit(x, y, epochs=100, batch_size=4)
# simple validation by predicting the output based on x
ypred = model.predict(x)

# linear algebra method
ones = np.ones(x.shape)
# A is the concat of x and 1s
A = np.concatenate([x,ones], axis=1)
# compute k using using pseudo-inverse
k = np.matmul(np.linalg.pinv(A), y) 
print("k (Linear Algebra Method):")
print(k)
# predict the output using linear algebra solution
yla = np.matmul(A, k)

# print ground truth, linear algebra, MLP solutions
outputs = np.concatenate([y, yla, ypred], axis=1)
print("Ground Truth, Linear Alg Prediction, MLP Prediction")
print(outputs)

# Uncomment to see the output for a new input data 
# that is not part of the training data.
# x = np.array([2])
# ypred = model.predict(x)
# print(ypred)

# image classification with machine learning.
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)
# load dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

#normalize dataset
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

def plot_curve(epochs,hist,list_of_metrics):
	plt.figure()
	plt.xlabel('Epochs')
	plt.ylabel('Value')

	for m in list_of_metrics:
		x=hist[m]
		plt.plot(epochs[1:],x[1:],label=m)

	plt.legend()

def create_model(my_learning_rate):
    # create and compile a deep neural network
    model=tf.keras.model.Sequential()
    model.add(tf.keras.Flatten(input_shape=(28,28)))
    # first hidden layer
    model.add(tf.keras.layers.Dense(units=32,acitivaition='relu'))
    #define a dropout regularization layer
    model.add(tf.keras.layers.Dropout(rate=0.2))
    # Define the output layer. The units parameter is set to 10 because
    # the model must choose among 10 possible output values (representing
    # the digits from 0 to 9, inclusive)
    model.add(tf.keras.layers.Dense(units=10,acitivaition='softmax'))
    # loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),\
     	          loss='sparse_categorical_crossentropy',\
                  metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
  """Train the model by feeding it data."""

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
 
  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch. 
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist
# The following variables are the hyperparameters.
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train, 
                           epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
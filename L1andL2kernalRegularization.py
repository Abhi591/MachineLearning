import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

train_df=pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv')
test_df=pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv')
#shuffle the dataset
train_df.reindex(np.random.permutation(train_df.index))

#normalize
train_df_mean=train_df.mean()
train_df_std=train_df.std()
train_df_norm=(train_df - train_df_mean)/train_df_std

test_df_mean=test_df.mean()
test_df_std=test_df.std()
test_df_norm=(test_df - test_df_mean)/test_df_std

# The following code creates a feature layer containing three features:
# latitude X longitude (a feature cross)
# median_income
# population

feature_columns=[]
resolution_in_Zs = 0.3  # 3/10 of a standard deviation.

# create a bucket feature column for latitude
latitude_as_numeric_col=tf.feature_column.numeric_column('latitude')

latitude_boundaries=list(np.arange(int(min(train_df_norm['latitude'])),\
	                               int(max(train_df_norm['latitude'])),\
	                               resolution_in_Zs))

latitude=tf.feature_column.bucketized_column(latitude_as_numeric_col,latitude_boundaries)

# create a bucket feature column for longitude
longitude_as_numeric_col=tf.feature_column.numeric_column('longitude')

longitude_boundaries=list(np.arange(int(min(train_df_norm['longitude'])),\
	                               int(max(train_df_norm['longitude'])),\
	                               resolution_in_Zs))

longitude=tf.feature_column.bucketized_column(longitude_as_numeric_col,longitude_boundaries)

# create a feature cross of latitude and longitude
latitude_x_longitude=tf.feature_column.crossed_column([latitude,longitude],\
	                                                   hash_bucket_size=100)

crossed_feature=tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

# Convert the list of feature columns into a layer that will later be fed into
# the model. 
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

def plot_curve(epochs,mse):
	# plot a curve loss vs epochs
	plt.figure()
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.plot(epochs,mse,label='Loss')
	plt.legend()
	plt.ylim([mse.min()*0.95,mse.max()*1.03])
	plt.show()

#defining deep neural network

def create_model(my_learning_rate,my_feature_layer):
	
	model=tf.keras.models.Sequential()
	model.add(my_feature_layer)
	
	# Describe the topography of the model by calling the tf.keras.layers.Dense
	# method once for each layer. We've specified the following arguments:
	# units specifies the number of nodes in this layer.
	# activation specifies the activation function (Rectified Linear Unit).
	# name is just a string that can be useful when debugging.
	
	# Define the first hidden layer with 20 nodes.
	model.add(tf.keras.layers.Dense(units=20,\
		                            activation='relu',\
		                            kernel_regularizer=tf.keras.regularizers.l2(l=0.04),\
		                            name='Hidden1'))

	# Define the second hidden layer with 12 nodes.
	model.add(tf.keras.layers.Dense(units=12,\
		                            activation='relu',\
		                            kernel_regularizer=tf.keras.regularizers.l2(l=0.04),\
		                            name='Hidden2'))

	#define the output layer 
	model.add(tf.keras.layers.Dense(units=1,name='Output'))

	model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),\
		          loss='mean_squared_error',\
		          metrics=[tf.keras.metrics.MeanSquaredError()])
	return model

def model_train(model, dataset, epochs, label_name,
                batch_size=None):
  """Train the model by feeding it data."""

  # Split the dataset into features and label.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True) 

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  hist = pd.DataFrame(history.history)
  mse = hist["mean_squared_error"]

  return epochs, mse

# hyperparameter
learning_rate=0.01
epochs=20
batch_size=1000
label_name='median_house_value'

my_model_final=create_model(learning_rate,my_feature_layer)

#train the model on the normalized training set
epochs,mse=model_train(my_model_final,train_df_norm,epochs,label_name,batch_size)
plot_curve(epochs,mse)

test_features={name:np.array(value) for name,value in test_df_norm.items()}
test_label=np.array(test_features.pop(label_name))

# evaluate the linear regression model against the test data
my_model_final.evaluate(x=test_features,y=test_label,batch_size=batch_size)
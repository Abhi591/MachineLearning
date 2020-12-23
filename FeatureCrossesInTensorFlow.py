import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

# California Housing Dataset.
train_df=pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv')
test_df=pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv')

# scaling median house value
train_df['median_house_value']=train_df['median_house_value']/1000
test_df['median_house_value']=test_df['median_house_value']/1000

# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))

feature_columns=[]

# create a numerical feature column to represent latitude
latitude=tf.feature_column.numeric_column('latitude')
feature_columns.append(latitude)
# create a numerical feature column to represent longitude
longitude=tf.feature_column.numeric_column('longitude')
feature_columns.append(longitude)
# converting the list of feature columns into layer 
fp_feature_layer=layers.DenseFeatures(feature_columns)

def build_model(learning_rate,feature_layer):

	model=tf.keras.models.Sequential()
	model.add(feature_layer)
	
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),\
		loss='mean_squared_error',\
		metrics=[tf.keras.metrics.RootMeanSquaredError()])
	
	return model

def train_model(model,dataset,epochs,batch_size,label_name):
	
	features={name:np.array(value) for name,value in dataset.items()}
	label=np.array(features.pop(label_name))
	
	history=model.fit(x=features,y=label,\
		epochs=epochs,batch_size=batch_size,shuffle=True)
	
	epochs=history.epoch
	# isolate the mean squared error for each epoch
	hist=pd.DataFrame(history.history)
	rmse=hist['root_mean_squared_error']
	
	return epochs,rmse

def plot_the_loss_curve(epoch,rmse):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Root Mean Squared Error')

	plt.plot(epoch,rmse,label='Loss')
	plt.legend()
	plt.ylim([rmse.min()*0.94,rmse.max()*1.05])
	plt.show()

learning_rate=0.05
epochs=30
batch_size = 100
label_name='median_house_value'

# compile the model
my_model=build_model(learning_rate,fp_feature_layer)
# training the model on the training set
epochs,rmse=train_model(my_model,train_df,epochs,batch_size,label_name)
plot_the_loss_curve(epochs,rmse)

# Evaluate the model on the test dataset
test_features={name:np.array(label) for name,label in test_df.items()}
test_label=np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features,y=test_label,batch_size=batch_size)


# No. Representing latitude and longitude as 
# floating-point values does not have much 
# predictive power.

#----------------------------------Method 2--------------------------------

#representing latitude and longitude in buckets 
# Each bin represents all the neighborhoods within a single degree. 
# For example, neighborhoods at latitude 35.4 and 35.8 are in the same bucket,
# but neighborhoods in latitude 35.4 and 36.2 are in different buckets.

resolution_in_degree=1.0

# list to hold the generated feature column
feature_columns=[]
# create a bucket feature column for latitude
latitude_as_a_numeric_column=tf.feature_column.numeric_column('latitude')
latitude_boundaries=list(np.arange(int(min(train_df['latitude'])),\
	                               int(max(train_df['latitude'])),\
	                               resolution_in_degree))
latitude=tf.feature_column.bucketized_column(latitude_as_a_numeric_column,\
	                                         latitude_boundaries)
feature_columns.append(latitude)

# create a bucket feature column for longitude
longitude_as_a_numeric_column=tf.feature_column.numeric_column('longitude')
longitude_boundaries=list(np.arange(int(min(train_df['longitude'])),\
	                               int(max(train_df['longitude'])),\
	                               resolution_in_degree))
longitude=tf.feature_column.bucketized_column(longitude_as_a_numeric_column,\
	                                         longitude_boundaries)
feature_columns.append(longitude)

# converting the list of feature column into layer
buckets_feature_layer=layers.DenseFeatures(feature_columns)

#train the model
#hyperparameters
learning_rate=0.04
epochs=35
# compile the model
my_model=build_model(learning_rate,buckets_feature_layer)
# training the model on the training set
epochs,rmse=train_model(my_model,train_df,epochs,batch_size,label_name)
plot_the_loss_curve(epochs,rmse)

# Evaluate the model on the test dataset
my_model.evaluate(x=test_features,y=test_label,batch_size=batch_size)
# Bucket representation outperformed 
# floating-point representations

#------------------------------Method 3 ------------------------------------

# Representing location as a feature cross should 
# produce better results.

resolution_in_degrees=0.4
#create a list which will hold the generated feature column
feature_columns=[]

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])), \
	                                 int(max(train_df['latitude'])), \
	                                 resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,\
                                               latitude_boundaries)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), \
	                                  int(max(train_df['longitude'])), \
	                                  resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, \
	                                            longitude_boundaries)

# create a feature cross of latitude and longitude
latitude_x_longitude=tf.feature_column.crossed_column([latitude,longitude],\
	                                                   hash_bucket_size=100)

crossed_feature=tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# convert the feature column into layer
feature_cross_feature_layer=layers.DenseFeatures(feature_columns)

#train the model
#hyperparameters
learning_rate=0.04
epochs=35
# compile the model
my_model=build_model(learning_rate,feature_cross_feature_layer)
# training the model on the training set
epochs,rmse=train_model(my_model,train_df,epochs,batch_size,label_name)
plot_the_loss_curve(epochs,rmse)

# Evaluate the model on the test dataset
my_model.evaluate(x=test_features,y=test_label,batch_size=batch_size)

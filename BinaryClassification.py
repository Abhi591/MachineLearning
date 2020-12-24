import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import feature_column
from tensorflow.keras import layers

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# tf.keras.backend.set_floatx('float32')

train_df=pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv')
test_df=pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv')
#shuffle the dataset
train_df.reindex(np.random.permutation(train_df.index))   

#normalize the dataset using z-score
#calculate the z-score of each column
# write those Z-scores into a new pandas DataFrame named train_df_norm.
train_df_mean=train_df.mean()
train_df_std=train_df.std()
train_df_norm=(train_df-train_df_mean)/train_df_std
# print(train_df_norm.head())

# Calculate the Z-scores of each column in the test set and
# write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

# a Z-score
# of +1.0 as the threshold, meaning that no more
# than 16% of the values in median_house_value_is_high
# will be labeled 1.

threshold_in_z=1.0
train_df_norm['median_house_value_is_high']=(train_df_norm['median_house_value'] > threshold_in_z ).astype(float)
test_df_norm['median_house_value_is_high']=(test_df_norm['median_house_value'] > threshold_in_z).astype(float)

# create an empty list to store the create feature columns
feature_columns=[]
median_income=tf.feature_column.numeric_column('median_income')
feature_columns.append(median_income)
total_rooms=tf.feature_column.numeric_column('total_rooms')
feature_columns.append(total_rooms)

# Convert the list of feature columns into a layer that will later be fed into
# the model.
featur_layer=layers.DenseFeatures(feature_columns)

def build_model(learning_rate,featur_layer,my_metrics):
	model=tf.keras.models.Sequential()
	model.add(featur_layer)
	# Funnel the regression value through a sigmoid function.
	model.add(tf.keras.layers.Dense(units=1,input_shape=(1,),activation=tf.sigmoid),)
	# we're using a different loss
	# function for classification than for regression.
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),\
		loss=tf.keras.losses.BinaryCrossentropy(),\
		metrics=my_metrics)

	return model

def train_model(model,dataset,epochs,label_name,batch_size=None,shuffle=True):
	features={name:np.array(value) for name,value in dataset.items()}
	label=np.array(features.pop(label_name))
	history=model.fit(x=features,y=label,\
		              epochs=epochs,batch_size=batch_size,\
		              shuffle=shuffle)
	epochs=history.epoch
	# Isolate the classification metric for each epoch.
	hist=pd.DataFrame(history.history)
	return epochs,hist

def plot_curve(epochs,hist,list_of_metrics):
	#classification metrics vs epochs
	plt.figure()
	plt.xlabel('Epochs')
	plt.ylabel('Value')

	for m in list_of_metrics:
		x=hist[m]
		plt.plot(epochs[1:],x[1:],label=m)
	plt.legend()
	plt.show()

# specify hyperparameters 
learning_rate=0.001
epochs=20
batch_size=100
label_name='median_house_value_is_high'
classification_threshold=0.52

# Establish the metrics
METRICS=[tf.keras.metrics.BinaryAccuracy(name='accuracy',\
	                                     threshold=classification_threshold),
	     tf.keras.metrics.Precision(name='precision',\
	     	                        thresholds=classification_threshold),
	     tf.keras.metrics.Recall(name='recall',\
	     	                     thresholds=classification_threshold),
	     tf.keras.metrics.AUC(name='auc',num_thresholds=100),]

my_model=build_model(learning_rate,featur_layer,METRICS)

#train the model
epochs,hist=train_model(my_model,train_df_norm,epochs,label_name,batch_size,)

# plot the graph
list_of_metrics_to_plot=['accuracy','precision','recall','auc']
plot_curve(epochs,hist,list_of_metrics_to_plot)

#testing the model on test dataset
test_features={name:np.array(value) for name,value in test_df_norm.items()}
test_label=np.array(test_features.pop(label_name))

my_model.evaluate(x=test_features,y=test_label,batch_size=batch_size)
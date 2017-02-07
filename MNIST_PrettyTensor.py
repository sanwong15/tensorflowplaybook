import matplotlib.pyplot as plt
import tensorflow as tf
import prettytensor as pt
import numpy as np
import math
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix

# Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

# Data Description
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# Class are 0 1 2 3 4 5.... 9
data.test.cls = np.argmax(data.test.labels,axis=1)

# Data Dimensions
img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 10

# Helper function to plot images
def plot_images(images, cls_true, cls_pred = None):
	assert len(images) == len(cls_true) == 9

	# Create 3 by 3 plot
	fig, axes = plt.subplots(3,3)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

	for i,ax in enumerate(axes.flat):
		#Plot images
		ax.imshow(images[i].reshape(img_shape),cmap='binary')

		#Show true and predicted num_classes
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i],cls_pred[i])

		ax.set_xlabel(xlabel)

		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

# Sample plot
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true = cls_true)

# Placeholder
x = tf.placeholder(tf.float32, shape=[None,img_size_flat], name = 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32,shape=[None,10],name='y_true')

# Create class label
y_true_cls = tf.argmax(y_true,dimension=1)

# TensorFlow Implementation
# Helper function:
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	# Shape of the filter weights for the conv layer
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	weights = new_weights(shape=shape)
	biases = new_biases(length=num_filters)

	layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
	layer += biases

	if use_pooling:
		layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	# layer = tf.nn.relu(layer)
	# Use sigmoid
	layer = tf.sigmoid(layer)

	return layer, weights


def flatten_layer(layer):
	layer_shape = layer.get_shape()

	#layer_shape == [num_images, img_height, img_width, num_channels]
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer,[-1,num_features])

	return layer_flat, num_features

def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
	weights = new_weights(shape=[num_inputs,num_outputs])
	biases = new_biases(length=num_outputs)

	layer = tf.matmul(input,weights)+biases

	if use_relu:
		# layer = tf.nn.relu(layer)
		# Use sigmoid
		layer = tf.sigmoid(layer)

	return layer
	
# PrettyTensor Implementation
# Wrap the input tensor x_image in a PrettyTensor Object
# that has helper function for adding new computational layers
# and hence create an entire CNN
x_pretty = pt.wrap(x_image)

# Do it as follow
# replace relu with sigmoid
# Use l2loss to decay weight
with pt.defaults_scope(activation_fn = tf.sigmoid):
	y_pred, loss = x_pretty.\
		conv2d(kernel=5,depth=16,l2loss=1,name='layer_conv1').\
		max_pool(kernel=2,stride=2).\
		conv2d(kernel=5,depth=36,l2loss=1,name='layer_conv2').\
		max_pool(kernel=2,stride=2).\
		flatten().\
		fully_connected(size=128,l2loss=1,name='layer_fc1').\
		dropout(0.5,name='dropout').\
		softmax_classifier(num_classes=num_classes,labels=y_true)

# How do we get the variable from each layer? We need to use get_variable from 
# TensorFlow library as everything in PrettyTensor is wrapped

def get_weights_variable(layer_name):
	with tf.variable_scope(layer_name, reuse=True):
		variable = tf.get_variable('weights')

	return variable

weights_conv1 = get_weights_variable(layer_name = 'layer_conv1')
weights_conv2 = get_weights_variable(layer_name = 'layer_conv2')	

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
# Boolean Array
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
# Convert boolean array
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# Run TensorFlow
session = tf.Session()
session.run(tf.global_variables_initializer())

# Optimization Iterations
train_batch_size = 64

# Keep track with iterations
total_iterations = 0

def optimize (num_iterations):
	global total_iterations

	start_time = time.time()

	for i in range(total_iterations, total_iterations+num_iterations):
		# Get data batch
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)

		feed_dict_train = {x:x_batch, y_true:y_true_batch}

		# Run optimizer using this training data batch
		session.run(optimizer,feed_dict = feed_dict_train)

		# Print for each 100 iteration
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict = feed_dict_train)

			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			print(msg.format(i + 1, acc))

	total_iterations += num_iterations
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
	incorrect = (correct == False)
	images = data.test.images[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = data.test.cls[incorrect]

	plot_images(images=images[0:9],cls_true=cls_true[0:9],cls_pred = cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
	# Need cls_true and cls_pred to calculatae Confusion Matrix
	cls_true = data.test.cls_pred
	cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
	print(cm)
	plt.matshow(cm)

	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks,range(num_classes))
	plt.yticks(tick_marks,range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')

	plt.show()


test_batch_size = 256

def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
	# To calculate accuracy, we need to know the number of images in the test-set
	num_test = len(data.test.images)

	# Create an array for the predicted classes
	cls_pred = np.zeros(shape=num_test,dtype=np.int)

	# Calculate the predicted classes for the batches
	i = 0

	while i<num_test:
		j = min(i+test_batch_size, num_test)
		images=data.test.images[i:j,:]
		labels = data.test.labels[i:j,:]

		feed_dict = {x: images, y_true: labels}

		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
		i=j

	cls_true = data.test.cls
	
	correct = (cls_true == cls_pred)

	correct_sum = correct.sum()
	acc = float(correct_sum)/num_test

	msg = "Accuracy on Test-set: {0:.1%} ({1}/{2})"
	print(msg.format(acc,correct_sum,num_test))

	# Plot some example of mis-classification
	if show_example_errors:
		print("Example errors")
		# def plot_example_errors(cls_pred, correct):
		plot_example_errors(cls_pred = cls_pred, correct = correct)

	if show_confusion_matrix:
		print("Confusion Matrix: ")
		# def plot_confusion_matrix(cls_pred):
		plot_confusion_matrix(cls_pred=cls_pred)

# Testing and Optimization
print("By gussing only, we got the following accuracy")
print_test_accuracy()

print("With one iteration on test batch")
optimize(num_iterations=1)
print_test_accuracy()

print("With 100 iteration on test batch")
optimize(num_iterations=99)
print_test_accuracy()

print("With 1000 iteration on test batch")
optimize(num_iterations=900)
print_test_accuracy()

print("With 10000 iteration on test batch")
optimize(num_iterations=9000)
print_test_accuracy()


# Visualization Weight and Layer
def plot_conv_weights(weights,input_channel = 0):
	w = session.run(weights)
	w_min = np.min(w)
	w_max = np.max(w)

		# def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
		# Shape of the filter weights for the conv layer
		# shape = [filter_size, filter_size, num_input_channels, num_filters]
		# weights = new_weights(shape=shape)

	num_filters = w.shape[3]
	num_grids = int (math.ceil(math.sqrt(num_filters)))
	fig, axes = plt.subplots(num_grids,num_grids)

	# Plot all the filter weights
	for i,ax in enumerate(axes.flat):
		if i<num_filters:
			img = w[:,:,input_channel,i]
			ax.imshow(img,vmin=w_min,vmax=w_max,interpolation='nearest',cmap='seismic')

		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()
	
print("Weight matrix of conv1")
plot_conv_weights(weights=weights_conv1)		


print("Weight matrix of conv2, first input channel. There are 16 input channel")
plot_conv_weights(weights=weights_conv2,input_channel=0)

print("Weight matrix of conv2, second input channel. There are 16 input channel")
plot_conv_weights(weights=weights_conv2,input_channel=1)

print("session close")
session.close()










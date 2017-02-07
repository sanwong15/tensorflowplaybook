import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Load Data
# In case you don't have these data, it will download it from tensorflow.exammples
from tensorflow.examples.tutorials.mnist import input_data
# We represent data in one_hot format
data = input_data.read_data_sets("data/MNIST/",one_hot = True)
print("Size of:")
print(" - Training-set:\t{}".format(len(data.train.labels)))
print(" - Test-set:\t\t{}".format(len(data.test.labels)))
print(" - Validation-set:\t{}".format(len(data.validation.labels)))

print("Print out the first 5 label of testing data")
data.test.labels[0:5,:]

data.test.cls = np.array([label.argmax() for label in data.test.labels])
print("Print out the first 5 class of testing data")
data.test.cls[0:5]


# Data dimension:
# test data = test sample size * Number of classes (i.e. n by 10 here)

# Var setting
img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 10

# PlOT helper function
def plot_images(images, true_class, predicted_class = None):
	assert len(images) == len(true_class) == 9 #Only run the followings if this is true

	#Create 3 by 3 subplot figure
	fig, axes = plt.subplots(3,3)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

	for i,ax in enumerate(axes.flat):
		# axes.flat returns the set of axes as a flat(ID) array
		#plot images
		ax.imshow(images[i].reshape(img_shape),cmap='binary')

		# Show True and Predicted classes
		if predicted_class is None:
			xlabel = "True: {0}".format(true_class[i])
		else:
			xlabel = "True: {0}, Predicted: {1}".format(true_class[i],predicted_class[i])

		ax.set_xlabel(xlabel)

		#Remove ticks
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()




train_image_shape = data.train.images.shape
test_image_shape = data.test.images.shape
validation_image_shape = data.validation.images.shape
print("train_image_shape:\t{}".format(train_image_shape)) 
print("test_image_shape:\t{}".format(test_image_shape))
print("validation_image_shape:\t{}".format(validation_image_shape)) 


# Test Plot function
images = data.test.images[0:9,:]
true_class = data.test.cls[0:9]
plot_images(images=images,true_class=true_class,predicted_class=None)


# Build the TensorFlow model/graph
# Helper function to randomly generate meright and biases
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))


# Helper function to create a Convolutional Layer
# 4-dim tensor
# (1) image number (2) Y-axis of each image, (3) X-axis of each image (4) Channels of each image

# Previous layer.
# Num of channel in previous layer
# Width and height of each folter
# Num of filters
# Use 2x2 max-pooling

def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):

	# shape of the filter-weights for the convolution
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# Create a new weights filter with the given shape
	weights = new_weights(shape = shape)
	biases = new_biases(length=num_filters)

	# Create the Tensorflow operation for convolution
	# The first and last strides are always 1
	# strides here is len 4 coz it is a 4-dim tensor. First one is for the image-number and 
	# last one is for input-channel
	# important example: if strides= [1,2,2,1] It means would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image. 
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same	

    	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    	layer += biases

    	#Use pooling to down sample
    	if use_pooling:
    		# As we are using 2by2 pooling windows, we can then move 2 pixels to the next window
    		layer = tf.nn.max_pool(value = layer, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

    # Rectified Linear Unit (ReLU) => Basically max(0,x)
    # This add non-linearity to the model -> allow it to learn some more complicated stuff

    	layer = tf.nn.relu(layer)

	return layer,weights

#Helper function to flattening a layer
# Why do we need this. At the fully connected layer after convolution layer, we need to 
# reduce the dimension 
def flatten_layer(layer):
	#Get the same of the input layer
	layer_shape = layer.get_shape()
	#It is assume that
	# layer_shape == [num_images, img_height, img_width, num_channels]
	# num of features is : img_height * img_width * num_channels
	num_features = layer_shape[1:4].num_elements()

	# Reshape the layer to [num_images, num_features]
	# Set the size as follow:
	# first dimension -> -1 (Here it means the size in this dimension is calculated)
	# second dimension -> num_features
	# Because first dimension is set to -1, the total size of the tensor is unchanged from reshapeing

	layer_flat = tf.reshape(layer,[-1, num_features])
	# Dimension of layer_flat: [num_images, img_height*img_width*num_channels]
	return layer_flat, num_features

# Helper function to create a new Fully-connected layer
# Four arguments
# (1) Previous layer
# (2) num of inputs from previous layer
# (3) num of outputs
# (4) use rectified linear unit
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	# Create new weights and biases
	weights = new_weights(shape=[num_inputs,num_outputs])
	biases = new_biases(length=num_outputs)

	# Calculate the pater as Matrix multiplication of input and weight and add biases
	layer = tf.matmul(input,weights) + biases

	if use_relu:
		layer = tf.nn.relu(layer)

	return layer




#Placeholder
x = tf.placeholder(tf.float32,[None, img_size_flat])
# We need to convert this x data into an image
# A four dimension tensor [num_images, img_height, img_width, num_channels]
# img_height == img_width == img_size. num_images can be inferred automatically by using -1
# for the size of the first dimension.
x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_true') # For true label
y_true_cls = tf.argmax(y_true, dimension=1) #For true class


# CNN configuration
# conv layer 1
filter_size1 = 5
num_filters1 = 16

# conv layer 2
filter_size2 = 5
num_filters2 = 36

# Fully-connected layer
fc_size = 128



# Create Convolutional Layer 1
# def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
# num_channels deault set to 1 at the begining of the code
layer_conv1, weights_conv1 = \
	new_conv_layer(input = x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling =True)

# Check the shape of the tensor by conv1. It should be (?,14,14,16). it means:
# ? = arbitrary number of images
# 14 = 14 pixels wide
# 14 = 14 pixel high
# 16 = 16 diff channel. One channel for each filter.
layer_conv1

# Create Convolutional Layer 2
layer_conv2,weights_conv2 = \
	new_conv_layer(input = layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
# Check the shape of the tensor by conv2. It should be (?,7,7,36)	
layer_conv2


# Flatten Layer
# The convolutional laters output 4-dim tensors. We now feed these output into a fully-connected layer
# However, before it is feed into a fully connected layer. It need to reshape or flattened
# def flatten_layer(layer):  return layer_flat, num_features

layer_flat, num_features = flatten_layer(layer_conv2)
# Check the dimension. should be (?, 1764) 1764 comes from 7*7*36
layer_flat
#Check num of features
num_features

# Fully Connected layer 1
# The number of neurons or nodes in the fully-connected layer is fc_size. 
# ReLU is used to learn non-linear relations
# def new_fc_layer(intput, num_inputs, num_outputs, use_relu=True): return layer
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

# Check the dimension of layer_fc1. Should be (?, 128)
layer_fc1

# Fully-connected layer 2
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=True)
# Check the dimension of layer_fc2. Should be (?,10)
layer_fc2


# Prediction with Softmax function (Class)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1) # class are 0,1,2,3,...9

# Cost-function to be optimized
# we have to use layer_fc2 (output directly from Fully connected layer 2, before it is plug into softmax)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimize with AdamOptimizer
# Try different values of Learning rate (1e-1, 1e-4 1e-10)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)


# Performance Measures
# correct_prediction is a vector of boolean whether the predicted class equals the true class of each image
correct_prediction = tf.equal(y_pred_cls,y_true_cls)

# accuracy then type cast the correct_prediction vector to floats. Flase = 0 and True = 1. Calculate the mean (which is accuracy here)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# Run TensorFlow
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
	# Update "total_iterations" be global
	global total_iterations
	start_time = time.time()

	for i in range(total_iterations, total_iterations+num_iterations):
		#Get a batch of training example
		#x_batch holds the image
		#y_true_batch holds the true label
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		#wrap x_batch and y_true_batch into a feed_dict
		feed_dict_train = {x:x_batch,y_true:y_true_batch}

		session.run(optimizer,feed_dict_train)



		# Print for every 100 iterations
		if i%100 == 0:
			# Calculate the accuracy on the training set
			acc = session.run(accuracy,feed_dict=feed_dict_train)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			#Print it
			print(msg.format(i+1,acc))

	# Update the total number of iterations performed
	total_iterations += num_iterations

	# End time
	end_time = time.time()

	# Time it takes
	time_diff = end_time - start_time

	#Print time usage
	print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))



def plot_example_errors(cls_pred, correct):
	
	
	incorrect = (correct == False)

	#Get the image from the test-set that was inocorrectly classified
	images = data.test.images[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = data.test.cls[incorrect]

	# plot_images(images, true_class, predicted_class = None):
	plot_images(images=images[0:9],
                true_class=cls_true[0:9],
                predicted_class=cls_pred[0:9])



def plot_confusion_matrix(cls_pred):
	#True classifications for the test-set
	cls_true = data.test.cls

	cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
	print(cm)

	plt.matshow(cm)

	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks,range(num_classes))
	plt.yticks(tick_marks,range(num_classes))
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.show()

#Split the test-set into smaller batches
test_batch_size = 256


def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
	#Number of image in the test-set
	num_test = len(data.test.images)

	# Allocate an array for the predicted classes 
	# which will be calculated in batches and filled into this array
	cls_pred = np.zeros(shape = num_test, dtype=np.int)

	# calculate the predicted class for the batches
	# initialize index i
	i = 0

	while i<num_test:
		# define the end index j
		# was thinking about j = i + test_batch_size. But this won't catch the concern case
		# a smarter way of doing this
		j = min(i+test_batch_size, num_test)

		# Get the images within the batch
		images = data.test.images[i:j,:]
		# Get the co-responding labels
		labels = data.test.labels[i:j,:]

		# Create a feed_dict as usual
		feed_dict = {x:images, y_true: labels}

		cls_pred[i:j] = session.run(y_pred_cls,feed_dict = feed_dict)

		# Update value of i for next batch
		i = j

	# To calculate the test-accuracy, with the cls_pred in hand. We still need cls_true
	# to check if we did good
	cls_true = data.test.cls

	# Create a boolean array
	correct = (cls_true == cls_pred)

	correct_sum = correct.sum()

	acc = float(correct_sum)/num_test

	msg = "Accuracy on Test-set: {0:.1%} ({1}/{2})"
	print(msg.format(acc,correct_sum,num_test))

	# Plot examples of mis-classification
	if show_example_errors:
		print("Example errors: ")
		# def plot_example_errors(cls_pred, correct):
		plot_example_errors(cls_pred = cls_pred, correct = correct)

	if show_confusion_matrix:
		print("Confusion matrix: ")
		# def plot_confusion_matrix(cls_pred):
		plot_confusion_matrix(cls_pred = cls_pred)


# By guessing, (use this as a base case to judge our CNN)
print("Test Accuracy by just guessing")
print_test_accuracy()

optimize(num_iterations=1)
print_test_accuracy(show_example_errors=True)

optimize(num_iterations=99)
print_test_accuracy(show_example_errors=True)

optimize(num_iterations=900)
print_test_accuracy(show_example_errors=True)

optimize(num_iterations=9000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)


# Plot weight (i.e weight_conv1 and weight_conv2)
def plot_conv_weights(weights, input_channel=0):
	w = session.run(weights)

	w_min = np.min(w)
	w_max = np.max(w)

	num_filters = w.shape[3]

	# Num of grids need for plotting
	num_grids = math.ceil(math.sqrt(num_filters))
	num_grids = int(num_grids)

	#Create fig
	fig, axes = plt.subplots(num_grids,num_grids)

	# Plot all weight
	for i,ax in enumerate(axes.flat):
		# Only plot the valid filter weights
		if i<num_filters:
			# Get the weights for the ith filter of the input channel
			img = w[:,:,input_channel,i]

			# plot image
			ax.imshow(img,vmin = w_min,vmax = w_max, interpolation = 'nearest', cmap = 'seismic')

		#Remove ticks
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

# Plot the output of convolution layer
# Each input data will have its own convultion layer result even given the same weight
def plot_conv_layer(layer, image):
	feed_dict = {x:[image]}

	# def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
	# return layer, weight

	# The value of the output of conv layer is store in tensor: layer
	values = session.run(layer, feed_dict = feed_dict)
	num_filters = values.shape[3] # Why is this true
	num_grids = math.ceil(math.sqrt(num_filters))
	num_grids = int(num_grids)

	# Create fig
	fig,axes = plt.subplots(num_grids,num_grids)

	for i,ax in enumerate(axes.flat):
		if i<num_filters:
			img = values[0,:,:,i]

			ax.imshow(img,interpolation='nearest',cmap='binary')

		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()	

# Plot only one image
def plot_image(image):
	plt.imshow(image.reshape(img_shape),interpolation='nearest',cmap='binary')
	plt.show()


image1 = data.test.images[0]
plot_image(image1)

image2 = data.test.images[1]
plot_image(image2)


# Plot Filter Weight
plot_conv_weights(weights=weights_conv1)
# Plot Convolution Layer 1
# def plot_conv_layer(layer, image):
plot_conv_layer(layer = layer_conv1, image = image1)
plot_conv_layer(layer = layer_conv1, image = image2)

# Plot Convolution Layer 2
plot_conv_weights(weights = weights_conv2)
plot_conv_layer(layer = layer_conv2, image = image1)
plot_conv_layer(layer = layer_conv2, image = image2)











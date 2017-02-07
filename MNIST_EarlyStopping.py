import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

# Placeholder
x = tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])

y_true = tf.placeholder(tf.float32,shape=[None,10],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)

x_pretty=pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
    conv2d(kernel=5,depth=16,name='layer_conv1').\
    max_pool(kernel=2,stride=2).\
    conv2d(kernel=5,depth=36,name='layer_conv2').\
    max_pool(kernel=2,stride=2).\
    flatten().\
    fully_connected(size=128,name='layer_fc1').\
    softmax_classifier(num_classes=num_classes,labels=y_true)

def get_weights_variable(layer_name):
    # Retrieve variable named 'weights' in the scope
    # with given layer_name

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Define which optimizer we use
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance Measures
y_pred_cls = tf.argmax(y_pred,dimension=1)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Define a SAVER to save variables of the neural network. Saver-object
saver = tf.train.Saver()
save_dir = 'checkpoints/'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Path for the checkpoint file
save_path = os.path.join(save_dir,'best_validation')

# Run TensorFlow
session = tf.Session()

def init_variables():
    session.run(tf.global_variables_initializer())

init_variables()

# Optimization Iterations
train_batch_size = 64

# Need a few variables to keep track of validation accuracy
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000

total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time = time.time()

    for i in range(num_iterations):
        total_iterations += 1

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict = feed_dict_train)

        if(total_iterations % 100 == 0) or (i==(num_iterations-1)):
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            # calculate the accuracy on the validation set
            # validation_accuracy returns 2 values.
            acc_validation, _ = validation_accuracy()

            # if validation accuracy improves
            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                # Mark down which iteration gives an improvement
                last_improvement = total_iterations
                # Save all variables of the TensorFlow graph to file
                saver.save(sess = session, save_path = save_path)
                # Show improvement found
                improved_str = '*'

            else:
                improved_str = ''

            # Status-message
            msg = 'Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%} {3}'

            print(msg.format(i+1,acc_train,acc_validation,improved_str))

        if total_iterations-last_improvement > require_improvement:
            print("No improvement found in a while. stop optimization")

            break


    end_time = time.time()

    time_diff = end_time-start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

def plot_example_errors(cls_pred, correct):
    incorrect = (correct==False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    # def plot_images(image,cls_true,cls_pred=None):
    # plot first 9
    plot_images(images = images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls

    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)

    print(cm)

    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Calculate Classification
batch_size = 256

# Find the predict classes and the co-responding Correct boolean array
def predict_cls(images,labels,cls_true):
    # number of images
    num_images = len(images)

    cls_pred = np.zeros(shape = num_images, dtype = np.int)

    i = 0

    while i<num_images:
        j = min(i+batch_size, num_images)

        feed_dict = {x: images[i:j,:], y_true: labels[i:j,:]}

        cls_pred[i:j] = session.run(y_pred_cls,feed_dict=feed_dict)

        i=j

    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = data.test.images, labels = data.test.labels, cls_true=data.test.cls)

def predict_cls_validation():
    return predict_cls(images = data.validation.images,labels = data.validation.labels,cls_true = data.validation.cls)


def cls_accuracy(correct):
    # input is an boolean array. Correct predict => 1 else 0
    correct_sum = correct.sum()
    acc = float(correct_sum)/len(correct)

    return acc, correct_sum


def validation_accuracy():
    # Get the array of boolean whether the classification are correct for validation set
    correct, _ = predict_cls_validation()

    return cls_accuracy(correct)

def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
    # For images in test-set, calculate the predicted class and check if they are correct
    # Use predict_cls_test to find the predicted class and boolean correct array
    correct, cls_pred = predict_cls_test()

    acc, num_correct = cls_accuracy(correct)

    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

print('Before Optimization')
print_test_accuracy()
plot_conv_weights(weights=weights_conv1)

print('With 10,000 Optimization iteration')
optimize(num_iterations=10000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1)


print('Initialize variables again')
init_variables()
print('Weight after Initialize')
plot_conv_weights(weights=weights_conv1)
print('Restore Best variables')
saver.restore(sess=session,save_path=save_path)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1)

print('Close session')
session.close()






































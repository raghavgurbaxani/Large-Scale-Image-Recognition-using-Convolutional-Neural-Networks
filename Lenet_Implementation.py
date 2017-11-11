import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

mnist=input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train=mnist.train.images, mnist.train.labels
X_validation, y_validation=mnist.validation.images, mnist.validation.labels
X_test, y_test=mnist.test.images, mnist.test.labels


# Pad images with 0s
X_train=np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation=np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test=np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

EPOCHS=10
BATCH_SIZE=128

#Function to define Weight variable
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Function to perform convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='VALID')
#Pooling Function
def max_pool(x):
    return tf.nn.max_pool(x,strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1] , padding='SAME')
#function for convolution layer
def conv_layer(input,shape):
    W=weight_variable(shape)
    b=bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input,W)+b)
#Function for fully connected layer
def full_layer(input,size):
    in_size=int(input.get_shape()[1])
    W=weight_variable([in_size,size])
    b=bias_variable([size])
    return tf.matmul(input,W)+b
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

conv_1=conv_layer(x, shape=[5, 5, 1, 6])     #sliding window of 5x5  and outputs a stack of 32 images
pool_1=max_pool(conv_1)

conv_2=conv_layer(pool_1, shape=[5, 5, 6, 16])   #sliding window of 5x5  and outputs a stack of 32 images
pool_2=max_pool(conv_2)

X_flat=flatten(pool_2)   #Flattening out the image

full_1=tf.nn.relu(full_layer(X_flat,120))

full_2=tf.nn.relu(full_layer(X_flat,84))

#SOFTMAX
logits=full_layer(full_2,10)

rate = 0.001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

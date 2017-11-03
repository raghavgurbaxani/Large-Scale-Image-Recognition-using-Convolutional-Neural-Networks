## Tensorflow Implementation of VGG 16 net

import tensorflow as tf

#Function for weight varaible
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Function to perform convolution
def conv2d(x,W,stride):
    return tf.nn.conv2d(x,W,strides=[1, stride, stride, 1], padding='SAME')

#Pooling Function
def max_pool(x,stride,k):
    return tf.nn.max_pool(x,strides=[1, stride, stride, 1], ksize=[1, k, k, 1] , padding='SAME')

#function for convolution layer
def conv_layer(input,shape,stride):
    W=weight_variable(shape)
    b=bias_variable([shape[3]])
    return tf.nn.relu(tf.nn.conv2d(input,W,strides=[1, stride, stride, 1],padding='SAME')+b)

#Function for fully connected layer
def full_layer(input,size):
    in_size=int(input.get_shape()[1])
    W=weight_variable([in_size,size])
    b=bias_variable([size])
    return tf.matmul(input,W)+b

# X_input is an image of shape 224x224x3

#Block 1
conv1=conv_layer(X_input, shape=[3, 3, 3, 64])                                  
conv2=conv_layer(conv1, shape=[3, 3, 64, 64])
pool_1=max_pool(conv2)

#Block 2                                                                                #The shape is now 112x112x64

conv3=conv_layer(pool_1, shape=[3, 3, 64, 128])
conv4=conv_layer(conv3, shape=[3, 3, 128, 128])
pool_2=max_pool(conv4)

#Block 3                                                                                ##The shape is now 56x56x128
                                                                                
conv5=conv_layer(pool_2, shape=[3, 3, 128, 256])
conv6=conv_layer(conv5, shape=[3, 3, 256, 256])
conv7=conv_layer(conv6, shape=[3, 3, 256, 256])
pool_3=max_pool(conv7)

#Block 4                                                                                #The shape is now 28x28x256

conv8=conv_layer(pool_3, shape=[3, 3, 256, 512])
conv9=conv_layer(conv8, shape=[3, 3, 512, 512])
conv10=conv_layer(conv9, shape=[3, 3, 512, 512])
pool_4=max_pool(conv10)

#Block 5                                                                                #The shape is now 14x14x512

conv11=conv_layer(pool_4, shape=[3, 3, 512, 512])
conv12=conv_layer(conv11, shape=[3, 3, 512, 512])
conv13=conv_layer(conv12, shape=[3, 3, 512, 512])
pool_5=max_pool(conv13)
                                                                                        #The output image is now 7x7 x 512
                                                                                
                                                                                
flat=tf.reshape(pool_5, [-1,7*7*512])                                                   #Flattening the output image to a 1d image

full_connected_1=tf.nn.relu(full_layer(flat,4096))                                       #Fully connected layer

full_connected_2=tf.nn.relu(full_layer(full_connected_1,4096))                           #Fully connected layer

full_connected_1=tf.nn.relu(full_layer(full_connected_2,NUM_CLASSES))                   #Softmax layer






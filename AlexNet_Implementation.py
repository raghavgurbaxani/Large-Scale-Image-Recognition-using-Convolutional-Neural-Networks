### Implementation of ALEXNET 

import tensorflow as tf

# function for weight variable
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Pooling Function
def max_pool(x,stride,k,padding='SAME'):
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

#Function for normalization
def LocalReluNorm(x, depth_radius, bias, alpha, beta):
	return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)



##   X_input is of shape 224x224x3
    


conv_1=conv_layer(X_input, shape=[11, 11, 3, 96],stride=4)                                              #Convolution layer
norm_1= LocalReluNorm(conv_1, depth_radius=2, bias=1.0,alpha=2e-05,beta=0.75)                           #Normalization layer
pool_1=max_pool(norm_1,stride=2,k=3,padding='VALID')                                                    # Pooling layer with valid pooling


conv_2=conv_layer(pool_1, shape=[5, 5, 96, 256],stride=1)
norm_2= LocalReluNorm(conv_2, depth_radius=2, bias=1.0,alpha=2e-05,beta=0.75)
pool_2=max_pool(norm_2,stride=2,k=3,padding='VALID')                                                    # Pooling layer with valid pooling


conv_3=conv_layer(pool_2, shape=[3, 3, 1, 384],stride=1)


conv_4=conv_layer(conv_3, shape=[3, 3, 1, 384],stride=1)


conv_5=conv_layer(conv_4, shape=[3, 3, 1, 256],stride=1)
pool_5=max_pool(conv_5,stride=2,k=3,padding='VALID')                                                    # Pooling layer with valid pooling


flat=tf.reshape(pool_5,[-1, 256 * 6 * 6])                                                               #Flattening out into a 1d vector

full_connected_1=full_layer(flat, 4096)                                                                 #Fully Connected Layer
full_drop_1=tf.nn.dropout(full_connected_1,keep_prob=dropout_probab1)                                   #Dropout on FC layer

full_connected_2=full_layer(full_connected_1, 4096)                                                     #Fully Connected layer
full_drop_2=tf.nn.dropout(full_connected_2,keep_prob=dropout_probab2)                                   #Dropout on FC layer

full_connected_3=full_layer(full_drop_2,NUM_CLASSES)                                                    #Softmax layer



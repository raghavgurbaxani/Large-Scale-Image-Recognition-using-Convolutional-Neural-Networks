## Tensorflow Implementation of Googlenet net

import tensorflow as tf

learning_rate=0.001
#Function for weight varaible
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Pooling Function
def max_pool(x,stride,k):
    return tf.nn.max_pool(x,strides=[1, stride, stride, 1], ksize=[1, k, k, 1] , padding='SAME')

#Pooling Function
def avg_pool(x,stride,k):
    return tf.nn.avg_pool(x,strides=[1, stride, stride, 1], ksize=[1, k, k, 1] , padding='SAME')

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

def local_response_norm(x,depth_radius=5, bias=1.0,alpha=0.0001,beta=0.75):
    return tf.nn.local_response_normalization(x)

def merge(tensors_list, axis=1):

    assert len(tensors_list) > 1, "Merge required 2 or more tensors."
    tensors = [l for l in tensors_list]
    inference = tf.concat(tensors, axis)
    return inference

X_input=None   # X_input is an image of shape shape=[None, 227, 227, 3]

conv1_7_7 = conv_layer(X_input,shape=[7, 7, 3, 64] ,stride=2)
pool1_3_3 = max_pool(conv1_7_7,k=3, strides=2)
pool1_3_3 = local_response_norm(pool1_3_3)
conv2_3_3_reduce = conv_layer(pool1_3_3,shape=[1, 1, 64, 64], stride=1)
conv2_3_3 = conv_layer(conv2_3_3_reduce, shape=[3, 3, 64, 192],stride=1)
conv2_3_3 = local_response_norm(conv2_3_3)
pool2_3_3 = max_pool(conv2_3_3, k=3, stride=2)

# 3a
inception_3a_1_1 = conv_layer(pool2_3_3, shape=[1, 1, 192, 64],stride=1)
inception_3a_3_3_reduce = conv_layer(pool2_3_3,shape=[1, 1, 192, 96],stride=1)
inception_3a_3_3 = conv_layer(inception_3a_3_3_reduce, shape=[3, 3, 96, 128],stride=1)
inception_3a_5_5_reduce = conv_layer(pool2_3_3,shape=[1, 1, 192, 16],stride=1)
inception_3a_5_5 = conv_layer(inception_3a_5_5_reduce,shape=[5, 5, 16, 32],stride=1)
inception_3a_pool = max_pool(pool2_3_3, k=3, stride=1)
inception_3a_pool_1_1 = conv_layer(inception_3a_pool,shape=[1, 1, 192, 32],stride=1)
inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], axis=3)
shape1=inception_3a_output.get_shape()[3]

# 3b
inception_3b_1_1 = conv_layer(inception_3a_output,shape=[1, 1, shape1, 128],stride=1)
inception_3b_3_3_reduce = conv_layer(inception_3a_output,shape=[1, 1, shape1, 128],stride=1)
inception_3b_3_3 = conv_layer(inception_3b_3_3_reduce,shape=[3, 3, 128, 192], stride=1)
inception_3b_5_5_reduce = conv_layer(inception_3a_output,shape=[1, 1, shape1, 32],stride=1)
inception_3b_5_5 = conv_layer(inception_3b_5_5_reduce,shape=[5, 5, 32, 96], stride=1)
inception_3b_pool = max_pool(inception_3a_output, k=3, stride=1)
inception_3b_pool_1_1 = conv_layer(inception_3b_pool,shape=[1, 1, shape1, 64], stride=1)
inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], axis=3)
pool3_3_3 = max_pool(inception_3b_output, k=3, stride=2)
shape2=pool3_3_3.get_shape()[3]
# 4a
inception_4a_1_1 = conv_layer(pool3_3_3,shape=[1, 1, shape2, 192],stride=1)
inception_4a_3_3_reduce = conv_layer(pool3_3_3,shape=[1, 1, shape2, 96], stride=1)
inception_4a_3_3 = conv_layer(inception_4a_3_3_reduce,shape=[3, 3, 96, 208], stride=1)
inception_4a_5_5_reduce = conv_layer(pool3_3_3,shape=[1, 1, shape2, 16], stride=1)
inception_4a_5_5 = conv_layer(inception_4a_5_5_reduce,shape=[5, 5, 16, 48] , stride=1)
inception_4a_pool = max_pool(pool3_3_3, k=3, stride=1)
inception_4a_pool_1_1 = conv_layer(inception_4a_pool,shape=[1, 1, shape2, 64],stride=1)
inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], axis=3)
shape3=inception_4a_output.get_shape()[3]

# 4b
inception_4b_1_1 = conv_layer(inception_4a_output, shape=[1, 1, shape3, 160],stride=1)
inception_4b_3_3_reduce = conv_layer(inception_4a_output,shape=[1, 1, shape3, 112],stride=1)
inception_4b_3_3 = conv_layer(inception_4b_3_3_reduce,shape=[3, 3, 112, 224],stride=1)
inception_4b_5_5_reduce = conv_layer(inception_4a_output,shape=[1, 1, shape3, 24],stride=1)
inception_4b_5_5 = conv_layer(inception_4b_5_5_reduce,shape=[5, 5, 24, 64],stride=1)
inception_4b_pool = max_pool(inception_4a_output, k=3, stride=1)
inception_4b_pool_1_1 = conv_layer(inception_4b_pool,shape=[1, 1, shape3, 64],stride=1)
inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], axis=3)
shape4=inception_4b_output.get_shape()[3]

# 4c
inception_4c_1_1 = conv_layer(inception_4b_output,shape=[1, 1, shape4, 128],stride=1)
inception_4c_3_3_reduce = conv_layer(inception_4b_output,shape=[1, 1, shape4, 128],stride=1)
inception_4c_3_3 = conv_layer(inception_4c_3_3_reduce,shape=[3, 3, 128, 256],stride=1)
inception_4c_5_5_reduce = conv_layer(inception_4b_output, shape=[1, 1, shape4, 24],stride=1)
inception_4c_5_5 = conv_layer(inception_4c_5_5_reduce,shape=[5, 5, 24, 64],stride=1)
inception_4c_pool = max_pool(inception_4b_output, k=3, stride=1)
inception_4c_pool_1_1 = conv_layer(inception_4c_pool,shape=[1, 1, shape4, 64],stride=1)
inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],axis=3)
shape5=inception_4c_output.get_shape()[3]

# 4d
inception_4d_1_1 = conv_layer(inception_4c_output, shape=[1, 1, shape5, 112],stride=1)
inception_4d_3_3_reduce = conv_layer(inception_4c_output,shape=[1, 1, shape5, 144],stride=1)
inception_4d_3_3 = conv_layer(inception_4d_3_3_reduce,shape=[3, 3, 144, 288],stride=1)
inception_4d_5_5_reduce = conv_layer(inception_4c_output,shape=[1, 1, shape5, 32],stride=1)
inception_4d_5_5 = conv_layer(inception_4d_5_5_reduce,shape=[5, 5, shape5, 64],stride=1)
inception_4d_pool = max_pool(inception_4c_output, k=3, stride=1)
inception_4d_pool_1_1 = conv_layer(inception_4d_pool,shape=[1, 1, shape5, 64],stride=1)
inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], axis=3)
shape6=inception_4d_output.get_shape()[3]

# 4e
inception_4e_1_1 = conv_layer(inception_4d_output,shape=[1, 1, shape6, 256],stride=1)
inception_4e_3_3_reduce = conv_layer(inception_4d_output,shape=[1, 1, shape6, 160],stride=1)
inception_4e_3_3 = conv_layer(inception_4e_3_3_reduce,shape=[3, 3, 160, 320],stride=1)
inception_4e_5_5_reduce = conv_layer(inception_4d_output,shape=[1, 1, shape6, 32],stride=1)
inception_4e_5_5 = conv_layer(inception_4e_5_5_reduce,shape=[5, 5, 32, 128],stride=1)
inception_4e_pool = max_pool(inception_4d_output, k=3, stride=1)
inception_4e_pool_1_1 = conv_layer(inception_4e_pool,shape=[1, 1, shape6, 128],stride=1)
inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3)
pool4_3_3 = max_pool(inception_4e_output, k=3, stride=2)
shape7=pool4_3_3.get_shape()[3]
# 5a
inception_5a_1_1 = conv_layer(pool4_3_3,shape=[1, 1, shape7, 256],stride=1)
inception_5a_3_3_reduce = conv_layer(pool4_3_3,shape=[1, 1, shape7, 160],stride=1)
inception_5a_3_3 = conv_layer(inception_5a_3_3_reduce,shape=[3, 3, 160, 320],stride=1)
inception_5a_5_5_reduce = conv_layer(pool4_3_3,shape=[1, 1, shape7, 32],stride=1)
inception_5a_5_5 = conv_layer(inception_5a_5_5_reduce,shape=[5, 5, 32, 128],stride=1)
inception_5a_pool = max_pool(pool4_3_3, k=3, stride=1)
inception_5a_pool_1_1 = conv_layer(inception_5a_pool,shape=[1, 1, shape7, 128],stride=1)
inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3)
shape8=inception_5a_output.get_shape()[3]


# 5b
inception_5b_1_1 = conv_layer(inception_5a_output,shape=[1, 1, shape8, 384],stride=1)
inception_5b_3_3_reduce = conv_layer(inception_5a_output,shape=[1, 1, shape8, 192],stride=1)
inception_5b_3_3 = conv_layer(inception_5b_3_3_reduce,shape=[3, 3, 192, 384],stride=1)
inception_5b_5_5_reduce = conv_layer(inception_5a_output,shape=[1, 1, shape8, 48],stride=1)
inception_5b_5_5 = conv_layer(inception_5b_5_5_reduce,shape=[5, 5, 48, 128],stride=1)
inception_5b_pool = max_pool(inception_5a_output, k=3, stride=1)
inception_5b_pool_1_1 = conv_layer(inception_5b_pool, shape=[1, 1, shape8, 128],stride=1)
inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3)
pool5_7_7 = avg_pool(inception_5b_output, k=7, stride=1)
pool5_7_7 = tf.nn.dropout(pool5_7_7, keep_prob=0.4)

# fc
loss = full_layer(pool5_7_7, 17)





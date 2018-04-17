import numpy as np
import tensorflow as tf
from model import model

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

def get_easy_conv(x, keep_prob):
    start_var = set(v.name for v in tf.global_variables())

    # x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

    end_var = tf.global_variables()
    new_var = [v for v in end_var if v.name not in start_var]

    # saver = tf.train.Saver()
    #
    # sess.run(tf.local_variables_initializer())
    #
    # saver.restore(sess, "easy_conv/")

    simple_conv = model(y_conv, cross_entropy, [x, keep_prob], y_)

    return simple_conv, new_var

def ConvMaxBloc(x, k, kernel, stride):
    output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, [2, 2], [2, 2])

    return output

def get_max_conv(x, keep_prob):
    start_var = set(v.name for v in tf.global_variables())

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    block1 = ConvMaxBloc(x_image, 32, [3, 3], [1, 1])
    block2 = ConvMaxBloc(block1, 64, [3, 3], [1, 1])
    blcok2 = tf.nn.dropout(block2, keep_prob)

    shape = block2.get_shape().as_list()
    output = tf.reshape(block2, [-1, shape[1]*shape[2]*shape[3]])
    output = tf.layers.dense(output, 100, tf.nn.relu)
    # output = tf.layers.dense(output, 100, tf.nn.relu)
    output = tf.nn.dropout(output, keep_prob)
    y_conv = tf.layers.dense(output, 10, tf.nn.relu)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

    end_var = tf.global_variables()
    new_var = [v for v in end_var if v.name not in start_var]

    max_conv = model(y_conv, cross_entropy, [x, keep_prob], y_)

    return max_conv, new_var

def ConvNormRelu(x, k, kernel, stride):
    output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
    output = tf.layers.batch_normalization(output)
    output = tf.nn.relu(output)
    return output

def get_BN_conv(x, keep_prob):
    start_var = set(v.name for v in tf.global_variables())

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    output = ConvNormRelu(x_image, 64, [8, 8], [1, 1])
    output = ConvNormRelu(output, 128, [6, 6], [1, 1])
    output = tf.nn.dropout(output, keep_prob)

    output = ConvNormRelu(output, 128, [4, 4], [1, 1])
    shape = output.get_shape().as_list()
    output = tf.reshape(output, [-1, shape[1]*shape[2]*shape[3]])
    output = tf.layers.dense(output, 50, tf.nn.relu)
    output = tf.nn.dropout(output, keep_prob)

    y_conv = tf.layers.dense(output, 10, tf.nn.relu)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

    end_var = tf.global_variables()
    new_var = [v for v in end_var if v.name not in start_var]

    BN_conv = model(y_conv, cross_entropy, [x, keep_prob], y_)

    return BN_conv, new_var

def get_naive_nn():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.get_variable(initializer = tf.zeros([784, 10]), name = 'softmax_W')
    b = tf.Variable(tf.zeros([10]), name = 'bias')

    # sess.run(tf.local_variables_initializer())

    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

    # saver = tf.train.Saver()
    # saver.restore(sess, "trial_save/")

    naive_nn = model(y, cross_entropy, [x], y_)
    return naive_nn

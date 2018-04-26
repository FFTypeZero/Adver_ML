import numpy as np
import tensorflow as tf
from cleverhans.model import Model

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

class easy_conv(Model):
    def __init__(self, keep_prob):
        super(easy_conv, self).__init__()

        self.keep_prob = keep_prob
        self.model_var = None
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

    def fprop(self, x):
        start_var = set(v.name for v in tf.global_variables())

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

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        prob = tf.nn.softmax(y_conv)

        end_var = tf.global_variables()
        new_var = [v for v in end_var if v.name not in start_var]
        self.model_var = new_var

        states = {"probs": prob, "logits": y_conv}
        return states

    def get_layer_names(self):

        return ["probs", "logits"]

    def predict(self, x):

        return self.fprop(x)["logits"]

def ConvMaxBloc(x, k, kernel, stride):
    output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, [2, 2], [2, 2])

    return output

class max_conv(Model):
    def __init__(self, keep_prob):
        super(max_conv, self).__init__()

        self.keep_prob = keep_prob
        self.model_var = None
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

    def fprop(self, x):
        start_var = set(v.name for v in tf.global_variables())

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        block1 = ConvMaxBloc(x_image, 32, [3, 3], [1, 1])
        block2 = ConvMaxBloc(block1, 64, [3, 3], [1, 1])
        blcok2 = tf.nn.dropout(block2, self.keep_prob)

        shape = block2.get_shape().as_list()
        output = tf.reshape(block2, [-1, shape[1]*shape[2]*shape[3]])
        output = tf.layers.dense(output, 100, tf.nn.relu)
        output = tf.nn.dropout(output, self.keep_prob)
        y_conv = tf.layers.dense(output, 10, tf.nn.relu)

        prob = tf.nn.softmax(y_conv)

        end_var = tf.global_variables()
        new_var = [v for v in end_var if v.name not in start_var]

        self.model_var = new_var

        states = {"probs": prob, "logits": y_conv}
        return states

    def get_layer_names(self):

        return ["probs", "logits"]

    def predict(self, x):

        return self.fprop(x)["logits"]

def ConvNormRelu(x, k, kernel, stride):
    output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
    output = tf.layers.batch_normalization(output)
    output = tf.nn.relu(output)
    return output

class BN_conv(Model):
    def __init__(self, keep_prob):
        super(BN_conv, self).__init__()

        self.keep_prob = keep_prob
        self.model_var = None
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

    def fprop(self, x):
        start_var = set(v.name for v in tf.global_variables())

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        output = ConvNormRelu(x_image, 64, [8, 8], [1, 1])
        output = ConvNormRelu(output, 128, [6, 6], [1, 1])
        output = tf.nn.dropout(output, self.keep_prob)

        output = ConvNormRelu(output, 128, [4, 4], [1, 1])
        shape = output.get_shape().as_list()
        output = tf.reshape(output, [-1, shape[1]*shape[2]*shape[3]])
        output = tf.layers.dense(output, 50, tf.nn.relu)
        output = tf.nn.dropout(output, self.keep_prob)

        y_conv = tf.layers.dense(output, 10, tf.nn.relu)

        prob = tf.nn.softmax(y_conv)

        end_var = tf.global_variables()
        new_var = [v for v in end_var if v.name not in start_var]

        self.model_var = new_var

        states = {"probs": prob, "logits": y_conv}
        return states

    def get_layer_names(self):

        return ["probs", "logits"]

    def predict(self, x):

        return self.fprop(x)["logits"]

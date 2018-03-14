import numpy as np
import tensorflow as tf

def ConvNormLRelu(x, name, k, leak_slope, kernel, stride, batch_norm, reuse):
    with tf.variable_scope(name, reuse = reuse):
        output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
        if batch_norm == True:
            output = tf.layers.batch_normalization(output)
        else:
            output = tf.contrib.layers.instance_norm(output)
        output = tf.nn.leaky_relu(output, leak_slope)
    return output

def D_mnist(x, kernel = [4, 4], stride = [2, 2], leak_slope = 0.2, batch_norm = False, reuse = False):
    start_var = set(x.name for x in tf.global_variables())

    output = ConvNormLRelu(x, "C8", 8, leak_slope, kernel, stride, batch_norm, reuse)
    output = ConvNormLRelu(output, "C16", 16, leak_slope, kernel, stride, batch_norm, reuse)
    output = ConvNormLRelu(output, "C32", 32, leak_slope, kernel, stride, batch_norm, reuse)

    shape = output.get_shape().as_list()
    output = tf.reshape(output, [-1, shape[1]*shape[2]*shape[3]])
    output = tf.layers.dense(output, 1, tf.nn.relu)

    end_var = tf.global_variables()
    new_var = [x for x in end_var if x.name not in start_var]

    return output, new_var

def test_D_mnist():
    trial_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    feed = np.zeros([25, 28, 28, 1])
    output, var_list = D_mnist(trial_input)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={trial_input: feed})
    print(result.shape)
    for x in var_list:
        print(x.name)

if __name__=='__main__':
    test_D_mnist()

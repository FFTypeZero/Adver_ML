import numpy as np
import tensorflow as tf

def ConvNormRelu(x, name, k, kernel, stride, batch_norm, reuse):
    with tf.variable_scope(name, reuse = reuse):
        output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
        if batch_norm == True:
            output = tf.layers.batch_normalization(output)
        else:
            output = tf.contrib.layers.instance_norm(output)
        output = tf.nn.relu(output)
    return output

def ResiBlock(x, name, k, kernel, stride, reuse):
    with tf.variable_scope(name, reuse = reuse):
        output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
        output = tf.nn.relu(output)
        output = tf.layers.conv2d(x, k, kernel, stride, padding='SAME')
        output = tf.nn.relu(output + x)
    return output

def DeConvNormRelu(x, name, k, kernel, stride, batch_norm, reuse):
    with tf.variable_scope(name, reuse = reuse):
        output = tf.layers.conv2d_transpose(x, k, kernel, stride, padding='SAME')
        if batch_norm == True:
            output = tf.layers.batch_normalization(output)
        else:
            output = tf.contrib.layers.instance_norm(output)
        output = tf.nn.relu(output)
    return output

def G_mnist(z, num_of_resi = 4, batch_norm = False, reuse = False):
    start_var = set(x.name for x in tf.global_variables())

    output = ConvNormRelu(z, "c3s1-8", 8, [3, 3], [1, 1], batch_norm, reuse)
    output = ConvNormRelu(output, "d16", 16, [3, 3], [2, 2], batch_norm, reuse)
    output = ConvNormRelu(output, "d32", 32, [3, 3], [2, 2], batch_norm, reuse)
    for i in range(num_of_resi):
        output = ResiBlock(output, "r32_{}".format(i+1), 32, [3, 3], [1, 1], reuse)
    output = DeConvNormRelu(output, "u16", 16, [3, 3], [2, 2], batch_norm, reuse)
    output = DeConvNormRelu(output, "u8", 8, [3, 3], [2, 2], batch_norm, reuse)
    output = ConvNormRelu(output, "c3s1-1", 1, [3, 3], [1, 1], batch_norm, reuse)

    end_var = tf.global_variables()
    new_var = [x for x in end_var if x.name not in start_var]

    output = tf.reshape(output, [-1, 28*28])

    return output, new_var

def test_G_mnist():
    trial_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    feed = np.zeros([25, 28, 28, 1])
    output, var_list = G_mnist(trial_input, 4)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={trial_input: feed})
    print(result.shape)
    for x in var_list:
        print(x.name)

if __name__=='__main__':
    test_G_mnist()

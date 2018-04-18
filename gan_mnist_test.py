import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from CW_attack import l2_attack
from get_model import get_easy_conv
from discriminator_mnist import D_mnist
from generator_mnist import G_mnist
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
z = tf.reshape(x, [-1, 28, 28, 1])

delta_x, G_var = G_mnist(z)
x_til = 0.5*(tf.tanh(delta_x/10) + 1)

keep_prob = tf.placeholder(tf.float32)
simple_conv, target_var = get_easy_conv(x_til, keep_prob)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list = target_var)
saver.restore(sess, "saved_models/easy_conv/")

y_ = tf.placeholder(tf.float32, [None, 10])
pred_op = simple_conv.get_pred()
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, tf.argmax(y_, axis=1)), tf.float32))

saver2 = tf.train.Saver(var_list = G_var)
saver2.restore(sess, "saved_models/G_easy/")

test_acc = []
for i in range(20):
    batch = mnist.test.next_batch(500)
    acc = sess.run(acc_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    test_acc.append(acc)
print("The test accuracy is {}.".format(np.mean(test_acc)))

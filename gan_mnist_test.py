import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from CW_attack import l2_attack
from get_model import get_easy_conv
from get_model import get_BN_conv
from get_model import get_max_conv
from discriminator_mnist import D_mnist
from generator_mnist import G_mnist
import os
import time

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'), cmap = 'gray')
            axs[i][j].axis('off')
    plt.savefig("trial.png")
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
z = tf.reshape(x, [-1, 28, 28, 1])

delta_x, G_var = G_mnist(z)
x_til = 0.5*(tf.tanh(delta_x/10) + 1)

# keep_prob = tf.placeholder(tf.float32)
# simple_conv, target_var = get_BN_conv(x_til, keep_prob)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver(var_list = target_var)
# saver.restore(sess, "saved_models/BN_conv/")

# y_ = tf.placeholder(tf.float32, [None, 10])
# pred_op = simple_conv.get_pred()
# acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, tf.argmax(y_, axis=1)), tf.float32))

saver2 = tf.train.Saver(var_list = G_var)
saver2.restore(sess, "saved_models/G_BN/")

# test_acc = []
# preds_num = []
adv_imgs = []
test_images = mnist.test.images
test_label = mnist.test.labels
start = time.time()
for i in range(20):
    batch = test_images[(i*500):((i+1)*500)]
    advs = sess.run(x_til, feed_dict={x: batch})
    adv_imgs.append(advs)
    # acc = sess.run(acc_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    # pred = sess.run(pred_op, feed_dict={x: batch[0], keep_prob: 1.0})
    # print(pred)
    # preds_num.append(np.sum(pred==3))
    # test_acc.append(acc)
end = time.time()

adv_imgs = np.array(adv_imgs)
np.savez("saved_advexamples/G_BN.npz", adv_imgs, test_label)
print(end - start)
# print("The test accuracy is {}.".format(np.mean(test_acc)))
# print(preds_num)

# test_images = mnist.test.images[0:100]
# adv_imgs = sess.run(x_til, feed_dict = {x: test_images})
# plot_digits(adv_imgs, 10, 10)
# print(np.linalg.norm(adv_imgs[0] - test_images[0]))

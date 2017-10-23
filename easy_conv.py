import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from sklearn import decomposition
from CW_attack import l2_attack

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

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'))
            axs[i][j].axis('off')
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
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

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.restore(sess, "easy_conv/")

simple_conv = model(y_conv, cross_entropy, train_step, [x, keep_prob])
test_images = mnist.test.images[0:100]
fd = [test_images, 1.0]
adv_img = l2_attack(simple_conv, sess, fd, 0.01, 3e-3, test_images, 3, 10)

# train_im = mnist.train.images
# train_pca = decomposition.PCA(50).fit(train_im)
# W = train_pca.components_
# adv_img_deno = adv_img @ W.T @ W

# logits = simple_conv.get_logits()
# result = logits.eval(feed_dict = {simple_conv.input:mnist.test.images}, session = sess)
# print(result.shape)
# for i in range(20000):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}, session = sess)
#         saver.save(sess, "easy_conv/")
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}, session = sess)
# saver.save(sess, "easy_conv/")

print("test accuracy %g" % accuracy.eval(feed_dict = {x: adv_img, y_: mnist.test.labels[0:100], keep_prob: 1.0}, session = sess))
plot_digits(adv_img, 10, 10)
plot_digits(adv_img_deno, 10, 10)
# print("test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.images[0:100], y_: mnist.test.labels[0:100], keep_prob: 1.0}, session = sess))

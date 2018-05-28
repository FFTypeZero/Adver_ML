import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from get_model import get_max_conv
from get_model import get_BN_conv

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'), cmap = 'gray')
            axs[i][j].axis('off')
    plt.show()

def binarize(x):
    mask_A = (x > 0.55).astype(np.float)
    mask_B = (x > 0.45).astype(np.float)
    return (mask_A + mask_B)/2.0

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
data = np.load("saved_advexamples/CW_max.npz")
imgs = data['arr_0']
labels = data['arr_1']
imgs = np.reshape(imgs, [10000, 784])
plot_digits(binarize(imgs[0:25]), 5, 5)

x = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder(tf.float32)
BN_conv, conv_var = get_max_conv(x, keep_prob)
cross_entropy = BN_conv.loss
y_ = BN_conv.label_ph
y_conv = BN_conv.logits

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "saved_models/max_conv/")

# for i in range(30000):
#     batch = mnist.train.next_batch(50)
#     sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#     if i % 100 == 0:
#         train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#         saver.save(sess, "saved_models/BN_conv/")
# saver.save(sess, "saved_models/BN_conv/")

# test_data = Batch_Creator(mnist.test)
test_acc = []
for i in range(20):
    # batch = mnist.test.next_batch(500)
    batch = (imgs[i*500:((i+1)*500)], labels[i*500:((i+1)*500)])
    bina_images = binarize(batch[0])
    acc = sess.run(accuracy, feed_dict={x: bina_images, y_: batch[1], keep_prob: 1.0})
    test_acc.append(acc)
print("The test accuracy is {}.".format(np.mean(test_acc)))
sess.close()

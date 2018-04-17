import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from get_model import get_max_conv
from get_model import get_BN_conv

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder(tf.float32)
BN_conv, conv_var = get_BN_conv(x, keep_prob)
cross_entropy = BN_conv.loss
y_ = BN_conv.label_ph
y_conv = BN_conv.logits

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
# saver.restore(sess, "saved_models/max_conv/")

for i in range(30000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        saver.save(sess, "saved_models/BN_conv/")
saver.save(sess, "saved_models/BN_conv/")

# test_data = Batch_Creator(mnist.test)
test_acc = []
for i in range(20):
    batch = mnist.test.next_batch(500)
    acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    test_acc.append(acc)
print("The test accuracy is {}.".format(np.mean(test_acc)))
sess.close()

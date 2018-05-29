from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Batch_Creator(object):
    def __init__(self, data):
        self.current_position = 0
        self.data = data[0]
        self.labels = data[1]
        self.size = len(data[0])

    def next_batch(self, batch_size):
        start = self.current_position
        end = self.current_position + batch_size
        if end >= self.size:
            self.current_position = 0
            return self.data[0: batch_size], np.reshape(self.labels[0: batch_size], batch_size)
        else:
            self.current_position = end
            return self.data[start: end], np.reshape(self.labels[start: end], batch_size)

#Parameters Initialization
PADDING_STRATEGY = 'SAME'
BATCH_SIZE = 100
TOTAL_SIZE = 50000
NUM_STEPS_PER_EPOCH = int(TOTAL_SIZE/BATCH_SIZE)
NUM_EPOCH = 200
SCHEDULE = np.array([200, 250, 300])
GAMMA = 0.00008
LAMBDA = 0.001
MOMENTUM = 0.9
PADDING_TENSOR = [[0, 0], [1, 1], [1, 1], [0, 0]]

#Data Preparation
train, test = tf.contrib.keras.datasets.cifar10.load_data()
cifar10_train = Batch_Creator(train)
cifar10_test = Batch_Creator(test)

#Model Assembly
if_drop = tf.placeholder(bool)
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int32, [None])
y_ = tf.one_hot(y, depth = 10)
in_drop = tf.contrib.layers.dropout(x, keep_prob=0.8, is_training = if_drop)
conv1 = tf.contrib.layers.conv2d(in_drop, num_outputs=96, kernel_size=[3, 3], padding = PADDING_STRATEGY)
conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=96, kernel_size=[3, 3], padding = PADDING_STRATEGY)
# conv2_pad = tf.pad(conv2, PADDING_TENSOR)
conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=96, kernel_size = [3, 3], stride = [2, 2], padding = PADDING_STRATEGY)
# conv3_pad = tf.pad(conv3, PADDING_TENSOR)
drop1 = tf.contrib.layers.dropout(conv3, keep_prob=0.5, is_training = if_drop)

conv4 = tf.contrib.layers.conv2d(drop1, num_outputs=192, kernel_size=[3, 3], padding = PADDING_STRATEGY)
# conv4_pad = tf.pad(conv4, PADDING_TENSOR)
conv5 = tf.contrib.layers.conv2d(conv4, num_outputs=192, kernel_size=[3, 3], padding = PADDING_STRATEGY)
# conv5_pad = tf.pad(conv5, PADDING_TENSOR)
conv6 = tf.contrib.layers.conv2d(conv5, num_outputs=192, kernel_size=[3, 3], stride = [2, 2], padding = PADDING_STRATEGY)
# conv6_pad = tf.pad(conv6, PADDING_TENSOR)
drop2 = tf.contrib.layers.dropout(conv6, keep_prob=0.5, is_training = if_drop)

conv7 = tf.contrib.layers.conv2d(drop2, num_outputs=192, kernel_size=[3, 3], padding = PADDING_STRATEGY)
# conv7_pad = tf.pad(conv7, PADDING_TENSOR)
conv8 = tf.contrib.layers.conv2d(conv7, num_outputs=192, kernel_size=[1, 1], padding = PADDING_STRATEGY)
conv9 = tf.contrib.layers.conv2d(conv8, num_outputs=10, kernel_size=[1, 1], padding = PADDING_STRATEGY)

global_ave_pool = tf.reduce_mean(tf.reduce_mean(conv9, axis=2), axis=1)

# Loss Function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = global_ave_pool)
tvars = tf.trainable_variables()
regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * LAMBDA
loss = cross_entropy + regularizer

#Optimizer
values = [GAMMA*(0.1**i) for i in range(len(SCHEDULE)+1)]
boundaries = list(NUM_STEPS_PER_EPOCH * SCHEDULE)
global_step = tf.Variable(0, trainable = False, dtype = tf.int64)
lr = tf.train.piecewise_constant(global_step, boundaries, values)
train_op = tf.train.MomentumOptimizer(learning_rate = lr, momentum = MOMENTUM).minimize(loss, global_step = global_step)
# train_op = tf.train.AdamOptimizer(learning_rate = GAMMA).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(global_ave_pool, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

#Model Training and Evaluation
ini_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(ini_op)
    # saver.restore(sess, "saved_models/all_conv/")
    test_accuracy = []
    for i in range(NUM_EPOCH * NUM_STEPS_PER_EPOCH):
        batchxs, batchys = cifar10_train.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict = {x: batchxs, y: batchys, if_drop: True})
        if i % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batchxs, y: batchys, if_drop: False})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            saver.save(sess, "saved_models/all_conv/")
        saver.save(sess, "saved_models/all_conv/")
    for j in range(20):
        testxs, testys = cifar10_test.next_batch(500)
        test_accuracy.append(accuracy.eval(feed_dict = {x: testxs, y: testys, if_drop: False}))
    print('The Test Accuracy is %g' % (np.mean(test_accuracy)))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from CW_attack import l2_attack
from get_model import get_easy_conv
from get_model import get_max_conv
from get_model import get_BN_conv
from discriminator_mnist import D_mnist
from generator_mnist import G_mnist
from get_model import get_all_conv
import os

BATCH_SIZE = 50
LEARNING_RATE = 1e-5
N_CRITIC = 5
MAX_ITERATION = 30000
LAMBDA = 10
ALPHA = 5
BETA = 1.0
GAMMA = 0.1

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

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, 32, 32, 3])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(data[i][j])
            axs[i][j].axis('off')
    plt.show()

def Loss_adv(model, target, confidence = 0, targeted = True):
    target_labels = tf.one_hot(target, depth = 10)
    lo = model.get_logits()
    other = tf.reduce_max((1 - target_labels)*lo - target_labels*10, axis=1)
    tar = tf.reduce_max(target_labels*lo, axis=1)
    if targeted:
        return tf.maximum(0.0, other - tar + confidence)
    else:
        return tf.maximum(0.0, tar - other + confidence)

def loss_simple_adv(model, target, targeted = True):
	target_labels = tf.one_hot(target, depth = 10)
	lo = model.get_logits()
	return -tf.reduce_max(target_labels*lo, axis=1)

train, test = tf.contrib.keras.datasets.cifar10.load_data()
cifar10_train = Batch_Creator(train)
cifar10_test = Batch_Creator(test)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
z = tf.placeholder(tf.float32, [None, 32, 32, 3])
if_drop = tf.placeholder(tf.bool)
simple_conv, target_var = get_all_conv(z, if_drop)

epsilon = tf.placeholder(tf.float32)

delta_x, G_var = G_mnist(x, 3)
x_til = 255*0.5*(tf.tanh(delta_x/10) + 1)
# x_til_imgs = tf.reshape(x_til, [-1, 28, 28, 1])


diff_x = tf.reshape(x_til - x, [-1, 32*32*3])
L_hinge = tf.maximum(0.0, tf.norm(diff_x, axis = 1))
diff_loss = tf.reduce_sum(L_hinge)/BATCH_SIZE
L_G = tf.reduce_sum(L_hinge)/BATCH_SIZE

update_G = tf.train.AdamOptimizer(LEARNING_RATE).minimize(L_G, var_list = G_var)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list = target_var)
saver.restore(sess, "saved_models/all_conv/")

y = tf.placeholder(tf.int32, [None])
y_ = tf.one_hot(y, depth = 10)
pred_op = simple_conv.get_pred()
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, tf.argmax(y_, axis=1)), tf.float32))

for f in os.listdir("graphs/gan_mnist_2"):
    os.remove("graphs/gan_mnist_2/"+f)
summary_writer = tf.summary.FileWriter("./graphs/gan_mnist_2", sess.graph)


tf.summary.scalar("G loss", L_G)
tf.summary.scalar("diff_loss", diff_loss)
tf.summary.image("adv_imgs", x_til)

summary_op = tf.summary.merge_all()
saver2 = tf.train.Saver(var_list = G_var)

for i in range(MAX_ITERATION):
    batchxs, batchys = cifar10_train.next_batch(BATCH_SIZE)
    _, summary = sess.run([update_G, summary_op], feed_dict = {x: batchxs})
    summary_writer.add_summary(summary, i)

    if i % 100 == 0:
        ae_imgs = sess.run(x_til, feed_dict = {x: batchxs})
        acc = sess.run([acc_op], feed_dict = {z: ae_imgs, y: batchys, if_drop: False})
        print("Step {}, target accuracy {}.".format(i+1, acc))
        # saver2.save(sess, "saved_models/AE_adv/")
# saver2.save(sess, "saved_models/AE_adv/")

test_images = test[0][0:100]
test_labels = test[1][0:100]
test_acc = sess.run(acc_op, feed_dict = {x: test_images, y_: test_labels, if_drop: False})
test_pred = sess.run(pred_op, feed_dict = {x: test_images, if_drop: False})
adv_imgs = sess.run(x_til, feed_dict = {x: test_images})
plot_digits(adv_imgs, 10, 10)
print(test_acc)
print(test_pred)

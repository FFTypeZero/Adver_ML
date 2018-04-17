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

BATCH_SIZE = 50
LEARNING_RATE = 1e-5
N_CRITIC = 5
MAX_ITERATION = 30000
LAMBDA = 10
ALPHA = 5
BETA = 17
GAMMA = 1.0

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'), cmap = 'gray')
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

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# test_images = mnist.test.images[0:100]
# plot_digits(test_images, 10, 10)

x = tf.placeholder(tf.float32, [None, 784])
# z = tf.placeholder(tf.float32, [None, 28, 28, 1])
z = tf.reshape(x, [-1, 28, 28, 1])
epsilon = tf.placeholder(tf.float32)

delta_x, G_var = G_mnist(z)
# x_til = delta_x + x
x_til = 0.5*(tf.tanh(delta_x/10) + 1)
x_til_imgs = tf.reshape(x_til, [-1, 28, 28, 1])
x_hat = epsilon * x + (1-epsilon) * x_til
Dx, D_var = D_mnist(x)
Dx_til = D_mnist(x_til, reuse = True)
Dx_hat = D_mnist(x_hat, reuse = True)

L_D = tf.reduce_sum(Dx_til - Dx + LAMBDA * (tf.norm(tf.gradients(Dx_hat, x_hat), axis=1) - 1)**2)/BATCH_SIZE

keep_prob = tf.placeholder(tf.float32)
simple_conv, target_var = get_easy_conv(x_til, keep_prob)

L_adv = Loss_adv(simple_conv, 3)
adv_loss = tf.reduce_sum(L_adv)/BATCH_SIZE
L_hinge = tf.maximum(0.0, tf.norm(x_til - x, axis=1) - 0.3)
diff_loss = tf.reduce_sum(L_hinge)/BATCH_SIZE
# L_hinge = 0
L_G = tf.reduce_sum(GAMMA*L_adv - ALPHA * Dx_til + BETA * L_hinge)/BATCH_SIZE

update_D = tf.train.AdamOptimizer(LEARNING_RATE).minimize(L_D, var_list = D_var)
update_G = tf.train.AdamOptimizer(LEARNING_RATE).minimize(L_G, var_list = G_var)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list = target_var)
saver.restore(sess, "saved_models/easy_conv/")

y_ = tf.placeholder(tf.float32, [None, 10])
pred_op = simple_conv.get_pred()
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, tf.argmax(y_, axis=1)), tf.float32))

for f in os.listdir("graphs/gan_mnist_2"):
    os.remove("graphs/gan_mnist_2/"+f)
summary_writer = tf.summary.FileWriter("./graphs/gan_mnist_2", sess.graph)

# tf.summary.scalar("D loss", L_D)
tf.summary.scalar("adv_loss", adv_loss)
tf.summary.scalar("G loss", L_G)
tf.summary.scalar("diff_loss", diff_loss)
tf.summary.image("adv_imgs", x_til_imgs)

summary_op = tf.summary.merge_all()

for i in range(MAX_ITERATION):
    for t in range(N_CRITIC):
        batch = mnist.train.next_batch(BATCH_SIZE)
        # z_feed = np.random.random_sample((BATCH_SIZE, 28, 28, 1))
        ep_feed = np.random.uniform()

        # sess.run(update_D, feed_dict = {x: batch[0], z: z_feed, epsilon: ep_feed})
        sess.run(update_D, feed_dict = {x: batch[0], epsilon: ep_feed})
    batch = mnist.train.next_batch(BATCH_SIZE)
    # z_feed_2 = np.random.random_sample((BATCH_SIZE, 28, 28, 1))
    # _, summary = sess.run([update_G, summary_op], feed_dict = {x: batch[0], z: z_feed_2, keep_prob: 1.0})
    _, summary = sess.run([update_G, summary_op], feed_dict = {x: batch[0], keep_prob: 1.0})
    summary_writer.add_summary(summary, i)

    # summary = sess.run(summary_op, feed_dict = {x: batch[0], z: z_feed, epsilon: ep_feed, keep_prob: 1.0})
    # summary_writer.add_summary(summary, i)
    if i % 100 == 0:
        # acc = sess.run(acc_op, feed_dict = {x: batch[0], z: z_feed_2, y_: batch[1], keep_prob: 1.0})
        acc = sess.run(acc_op, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("This is {}th training iteration and the target accuracy is {}.".format(i+1, acc))


test_images = mnist.test.images[0:100]
test_labels = mnist.test.labels[0:100]
# z_test = np.random.random_sample((100, 28, 28, 1))
# test_acc = sess.run(acc_op, feed_dict = {x: test_images, z: z_test, y_: test_labels, keep_prob: 1.0})
# adv_imgs = sess.run(x_til, feed_dict = {x: test_images, z: z_test})
test_acc = sess.run(acc_op, feed_dict = {x: test_images, y_: test_labels, keep_prob: 1.0})
adv_imgs = sess.run(x_til, feed_dict = {x: test_images})
plot_digits(adv_imgs, 10, 10)
print(np.amax(adv_imgs))
print(np.amin(adv_imgs))

#
# pred_op = simple_conv2.get_pred()
# pred = sess.run(pred_op, feed_dict = dict((zip([x, keep_prob], [test_images, 1.0]))))
# accu = np.mean(np.equal(pred, np.argmax(test_labels, axis=1)))
#
# print(test_images.shape)
# print(accu)

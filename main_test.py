import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from sklearn import decomposition
from CW_attack import l2_attack
from get_model import get_easy_conv
from get_model import get_max_conv
from get_model import get_BN_conv
from get_model import get_naive_nn
import time

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'), cmap = 'gray')
            axs[i][j].axis('off')
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x = tf.placeholder(tf.float32, [250, 784])
keep_prob = tf.placeholder(tf.float32)
sess = tf.Session()
simple_conv, _ = get_BN_conv(x, keep_prob)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "saved_models/BN_conv/")

preds2 = simple_conv.get_pred()

adv_imgs = []
test_images = mnist.test.images
# test_images = np.random.random_sample([100, 784])
test_labels = mnist.test.labels
train_op, newimg, loss = l2_attack(simple_conv, sess, 10, 3e-3, x, 3, 10)
start = time.time()
for i in range(40):
    batch = test_images[(i*250):((i+1)*250)]
    fd = [batch, 1.0]
    old_val = np.zeros(batch.shape)
    for j in range(3000):
        sess.run(train_op, feed_dict = dict(zip([x, keep_prob], fd)))
        retval = sess.run(newimg)
        loss_val = sess.run(loss, feed_dict = dict(zip([x, keep_prob], fd)))
        change = np.linalg.norm(old_val - retval)/np.linalg.norm(retval)
        old_val = retval
        if j % 100 == 0:
            print("This is {}th iteration. The change is {}.".format(j, np.linalg.norm(change)))
        if change < 1e-4:
            break
    adv_img = retval
    # predictions2 = sess.run(preds2, feed_dict=dict((zip(simple_conv.ph_list, [adv_img, 1.0]))))
    # accuracy2 = np.mean(np.equal(predictions2, np.argmax(test_labels[(i*500):((i+1)*500)], axis=1)))
    # print(accuracy2)
    adv_imgs.append(adv_img)
    print("{}th round.".format(i+1))
end = time.time()

adv_imgs = np.array(adv_imgs)
np.savez("saved_advexamples/CW_BN.npz", adv_imgs, test_labels)
print(end - start)
# preds2 = simple_conv.get_pred()
# prob = simple_conv.get_prob()


# predictions2 = sess.run(preds2, feed_dict=dict((zip(simple_conv.ph_list, [adv_img, 1.0]))))
# probability = sess.run(prob, feed_dict=dict((zip(simple_conv.ph_list, [adv_img, 1.0]))))
# accuracy2 = np.mean(np.equal(predictions2, np.argmax(test_labels, axis=1)))
# print("simple conv: test accuracy %g" % accuracy2)
# print(predictions2)
# plot_digits(adv_img, 10, 10)
# print(np.max(probability, axis=1))
sess.close()

tf.reset_default_graph()

# sess = tf.Session()
# naive_nn = get_naive_nn()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(sess, "saved_models/trial_save/")
#
# preds = naive_nn.get_pred()
# predictions = sess.run(preds, feed_dict=dict((zip(naive_nn.ph_list, [adv_img]))))
# accuracy = np.mean(np.equal(predictions, np.argmax(test_labels, axis=1)))
#
# print("naive nn: test accuracy %g" % accuracy)
# print(predictions)
# plot_digits(adv_img[0:49], 7, 7)

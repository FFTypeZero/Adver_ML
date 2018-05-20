import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# from cleverhans_model import easy_conv
# from cleverhans_model import max_conv
# from cleverhans_model import BN_conv
from get_model import get_easy_conv
from get_model import get_max_conv
from get_model import get_BN_conv
import os
from collections import Counter

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'), cmap = 'gray')
            axs[i][j].axis('off')
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
clean = mnist.test.images

data = np.load("saved_advexamples/G_easy.npz")
imgs = data['arr_0']
imgs = np.reshape(imgs, [10000, 784])
# labs = data['arr_1']
# positions = [[] for i in range(10)]
# chosen_data = []
# chosen_clean = []
# for i in range(len(labs)):
#     y = np.argmax(labs[i])
#     positions[y].append(i)
# for pos in positions:
#     for j in range(10):
#         chosen_data.append(imgs[pos[j]])
#         chosen_clean.append(clean[pos[j]])
# # plot_digits(np.array(chosen_data), 10, 10)
# plot_digits(chosen_clean, 10, 10)

SIZE = 500
# names = os.listdir("saved_advexamples/")
# paths = ["saved_advexamples/"+name for name in names]
# adv_examples = [np.load(path) for path in paths]

# mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# model = easy_conv(1.0)
simple_conv, target_var = get_max_conv(x, keep_prob)
probs = simple_conv.get_prob()
pred_op = tf.argmax(probs, axis = 1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, tf.argmax(y_, axis=1)), tf.float32))
# target_var = model.model_var

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list = target_var)
saver.restore(sess, "saved_models/max_conv/")

batchx = imgs[0:25]
batch_pred = sess.run(pred_op, feed_dict={x: batchx, keep_prob: 1.0})
print(batch_pred)
plot_digits(batchx, 10, 10)

# clean_images = mnist.test.images
# clean_labels = mnist.test.labels
# accs = []
# for j in range(3):
#     acc = []
#     preds = []
#     adv_imags = adv_examples[j]['arr_0']
#     labels = adv_examples[j]['arr_1']
#     for i in range(20):
#         batchx = adv_imags[i]
#         batchy = labels[(i*SIZE):((i+1)*SIZE)]
#         batch_pred = sess.run(pred_op, feed_dict={x: batchx, keep_prob: 1.0})
#         preds.extend(list(batch_pred))
#     print(Counter(preds).most_common())
# print(names)
# print(accs)

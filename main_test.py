import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model
from sklearn import decomposition
from CW_attack import l2_attack
from get_model import get_easy_conv
from get_model import get_naive_nn

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'))
            axs[i][j].axis('off')
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

sess = tf.Session()
simple_conv = get_easy_conv(sess)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "saved_models/easy_conv/")

test_images = mnist.test.images[0:100]
test_labels = mnist.test.labels[0:100]
fd = [test_images, 1.0]
adv_img = l2_attack(simple_conv, sess, fd, 0.01, 3e-3, test_images, 3, 10)
preds2 = simple_conv.get_pred()

predictions2 = sess.run(preds2, feed_dict=dict((zip(simple_conv.ph_list, [adv_img, 1.0]))))
accuracy2 = np.mean(np.equal(predictions2, np.argmax(test_labels, axis=1)))
print("simple conv: test accuracy %g" % accuracy2)
print(predictions2)
sess.close()

tf.reset_default_graph()

sess = tf.Session()
naive_nn = get_naive_nn()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "saved_models/trial_save/")

preds = naive_nn.get_pred()
predictions = sess.run(preds, feed_dict=dict((zip(naive_nn.ph_list, [adv_img]))))
accuracy = np.mean(np.equal(predictions, np.argmax(test_labels, axis=1)))

print("naive nn: test accuracy %g" % accuracy)
print(predictions)
plot_digits(adv_img, 10, 10)

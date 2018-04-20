import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans_model import easy_conv
from cleverhans_model import max_conv
from cleverhans_model import BN_conv

def plot_digits(vecs, nrows, ncols):
    data = np.reshape(vecs, [nrows, ncols, -1])
    f, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(np.reshape(data[i][j], [28, 28], order = 'C'), cmap = 'gray')
            axs[i][j].axis('off')
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x = tf.placeholder(tf.float32, [None, 784])

fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

model = max_conv(0.5)
states = model.fprop(x)
target_var = model.model_var
sess = tf.Session()

fgsm = FastGradientMethod(model, sess=sess)
adv_x = fgsm.generate(x, **fgsm_params)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list = target_var)
saver.restore(sess, "saved_models/max_conv/")

test_images = mnist.test.images[0:100]
adv_imgs = sess.run(adv_x, feed_dict={x: test_images})
plot_digits(adv_imgs, 10, 10)

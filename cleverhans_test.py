import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import MadryEtAl
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans_model import easy_conv
from cleverhans_model import max_conv
from cleverhans_model import BN_conv
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
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
SIZE = 200

tar = tf.one_hot(np.array([3 for i in range(SIZE)]), depth=10)
# fgsm_params = {'batch_size': 100, 'y_target': tf.one_hot(tar, depth = 10), 'clip_min': 0., 'clip_max': 1.}
# fgsm_params = {'clip_min': 0., 'clip_max': 1., 'eps': 0.3}

model = BN_conv(0.5)
probs = model.get_probs(x)
pred = tf.argmax(probs, axis = 1)
target_var = model.model_var
sess = tf.Session()
target = sess.run(tar)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list = target_var)
saver.restore(sess, "saved_models/BN_conv/")

fgsm_params = {'batch_size': 100, 'y_target': target, 'clip_min': 0., 'clip_max': 1., 'learning_rate': 1e-2, 'confidence': 3, 'max_iterations': 2000}
start_var = set(v.name for v in tf.global_variables())

# fgsm = FastGradientMethod(model, sess=sess)
# fgsm = BasicIterativeMethod(model, sess=sess)
# fgsm = MomentumIterativeMethod(model, sess=sess)
# fgsm = SaliencyMapMethod(model, sess=sess)
# fgsm = CarliniWagnerL2(model, sess=sess)
# fgsm = MadryEtAl(model, sess=sess)
# adv_x = fgsm.generate(x, **fgsm_params)
part = tf.diag_part(tf.matmul(probs, tf.transpose(y)))
adv_x = x + 0.2*tf.sign(tf.squeeze(tf.gradients(part, x)))

end_var = tf.global_variables()
new_var = [v for v in end_var if v.name not in start_var]

sess.run(tf.variables_initializer(var_list = new_var))


adv_imgs = []
test_images = mnist.test.images
test_label = mnist.test.labels
start = time.time()
for i in range(50):
    batch = test_images[(i*SIZE):((i+1)*SIZE)]
    advs = sess.run(adv_x, feed_dict={x: batch, y: target})
    adv_imgs.append(advs)
end = time.time()

adv_imgs = np.array(adv_imgs)
np.savez("saved_advexamples/FGSM_BN.npz", adv_imgs, test_label)
print(end - start)
print(adv_imgs.shape)

# adv_imgs = np.array(adv_imgs)
# np.savez("saved_advexamples/G_BN.npz", adv_imgs, test_label)
# print(end - start)
#
# test_images = mnist.test.images[0:SIZE]
# test_labels = mnist.test.labels[0:SIZE]
# adv_imgs = sess.run(adv_x, feed_dict={x: test_images, y: target})
# predictions = sess.run(pred, feed_dict={x: adv_imgs})
# preds = sess.run(pred, feed_dict={x: test_images})
# acc = np.mean(predictions == np.argmax(test_labels, axis = 1))
# clean_acc = np.mean(preds == np.argmax(test_labels, axis = 1))
# # plot_digits(adv_imgs, 10, 10)
# plot_digits(test_images, 10, 10)
# print(predictions)
# print(acc)
# print(clean_acc)
# print(np.linalg.norm(adv_imgs[0]-test_images[0]))

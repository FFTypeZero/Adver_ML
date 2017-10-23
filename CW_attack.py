import numpy as np
import tensorflow as tf

def obj_func(model, target_labels, confidence, targeted):
    lo = model.get_logits()
    other = tf.reduce_max((1 - target_labels)*lo - target_labels*10, axis=1)
    tar = tf.reduce_max(target_labels*lo, axis=1)
    if targeted:
        return tf.maximum(0.0, other - tar + confidence)
    else:
        return tf.maximum(0.0, tar - other + confidence)

def l2_attack(model, sess, fd, cons, lr, X, target, num_labels, confidence = 0, targeted = True, max_ite = 2000, tol = 1e-3):
    onehot_t = tf.one_hot(target, depth = num_labels)
    shape = X.shape
    # sum_idx = list(shape[1:len(shape)])
    w = tf.Variable(tf.zeros(shape), dtype = tf.float32)
    newimg = tf.tanh(w)/2 + 1
    loss_term2 = cons * obj_func(model, onehot_t, confidence, targeted)
    loss_term1 = tf.reduce_sum(tf.square(newimg - X), axis=1)
    loss = loss_term1 + loss_term2
    old_val = np.zeros(shape)

    train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[w])
    sess.run(tf.global_variables_initializer())
    for i in range(max_ite):
        sess.run(train_op, feed_dict = dict(zip(model.ph_list, fd)))
        retval = sess.run(newimg)
        loss_val = sess.run(loss, feed_dict = dict(zip(model.ph_list, fd)))
        change = np.linalg.norm(old_val - retval)/np.linalg.norm(retval)
        if i % 10 == 0:
            print("This is {}th iteration. The loss is {}.".format(i, np.linalg.norm(loss_val)))
        if change < tol:
            break
    return retval

import numpy as np
import tensorflow as tf

class model(object):

    def __init__(self, logits, loss, ph, label):
        self.logits = logits
        self.loss = loss
        self.ph_list = ph
        self.label_ph = label

    def get_logits(self):
        return self.logits

    def get_prob(self):
        return tf.nn.softmax(self.logits)

    def get_pred(self):
        return tf.argmax(self.logits, axis=1)

    def get_loss(self):
        return self.loss

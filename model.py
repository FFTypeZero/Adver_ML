import numpy as np
import tensorflow as tf

class model(object):

    def __init__(self, logits, loss, train_op, ph):
        self.logits = logits
        self.loss = loss
        self.train_op = train_op
        self.ph_list = ph

    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss

    def train(self, sess):
        sess.run(self.train_op)

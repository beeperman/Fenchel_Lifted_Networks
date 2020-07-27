import  tensorflow as tf
from nn import activations


class XFINInterface(object):
    """
    closed form solution for a special case (original LNN, regression problem, additional variables in the final layer)
    """
    def __init__(self, layer):
        self.layer = layer

        self.X_update_op = tf.no_op()

        if self.layer.X_initializer is not None:
            if self.layer.is_last_layer:
                y_one_hot = self.layer.X_next
                U = self.layer.prev_layer.penalty
                alpha = self.layer.rho
                gamma = self.layer.prev_layer.lmbda
                V = (y_one_hot + gamma * U) / (1.0 + gamma)
            else:
                U_1 = tf.matrix_transpose(activations[self.layer.prev_layer.activation](self.layer.prev_layer.U))
                U_2 = tf.matrix_transpose(self.layer.U)
                W_p = self.layer.W
                W = tf.matrix_transpose(self.layer.W)
                d = W.shape.as_list()[-1]
                b = self.layer.b[..., None]
                #gamma = self.layer.lmbda
                V = tf.matrix_transpose(tf.matmul(tf.matrix_inverse(tf.matmul(W_p, W) + tf.eye(d)), tf.matmul(W_p, U_2 - b) + U_1))

            self.X_update_op = self.layer.X.assign(V)


    def minimize(self, sess, feed_dict):
        sess.run(self.X_update_op, feed_dict)
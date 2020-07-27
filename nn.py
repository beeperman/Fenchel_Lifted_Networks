## Modified and updated from Corey's code
## Tensorflow operations

import tensorflow as tf
from tensorflow.contrib.layers import flatten, avg_pool2d
from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np

def f_norm_sq(X):
    """ Compute Frobenius Norm square """
    return tf.reduce_sum(tf.square(X))

def max_pool(X):
    """ 2x2 -> 1x1 max pooling"""
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def avg_pool(X):
    """ 2x2 -> 1x1 max pooling"""
    return avg_pool2d(X, kernel_size=2, stride=2, padding='VALID')


def l2(X_next, penalty):
    return tf.reduce_sum(tf.square(X_next - penalty))#, axis=list(range(1, len(X_next.shape)))))

def neo(X_next, penalty):
    #alpha = 0.0001
    u = penalty
    v = X_next
    u_plus = tf.clip_by_value(penalty, 0, np.inf)
    #return tf.reduce_sum(tf.square(v) - 2.0 * tf.multiply(u, v) + alpha * tf.square(u) + (1.0 - alpha) * tf.square(u_plus))
    return tf.reduce_sum(tf.square(v) - 2.0 * tf.multiply(u, v) + tf.square(u_plus))


def softmax_cross_entropy(labels, prediction):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))

def ll2(X_next, penalty):
    return tf.reduce_mean(tf.reduce_sum(tf.square(X_next - penalty), axis=list(range(1, len(X_next.shape)))))

def lsoftmax_cross_entropy(labels, prediction):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))

lnn_activation_ops = {"relu": l2, "cross_entropy": softmax_cross_entropy, "none": l2, 'none_ce': l2}
lnn_activation_ops_new = {"relu": neo, "cross_entropy": softmax_cross_entropy, "none": l2, 'none_ce':l2}
lnn_loss_ops = {"relu": ll2, "cross_entropy": lsoftmax_cross_entropy, "none": l2}
activations = {"relu": tf.nn.relu, "cross_entropy": tf.nn.softmax_cross_entropy_with_logits_v2, "none": tf.identity}


# wrapper for tensorflow optimizers

# gradient clipping
def minimize_and_clip(optimizer, objective, var_list, clip_val=10.0):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients), gradients

class ADAMOptimizerInterface(object):
    def __init__(self,
                 loss,
                 var_list=None,
                 lr=1e-3,
                 clip_val=10.0,
                 iteration=500,
                 **optimizer_kwargs):

        self.loss = loss
        self.var_list = var_list
        self.clip_val = clip_val
        self.iteration = iteration

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, **optimizer_kwargs)
        self.train_op, self.grad = minimize_and_clip(self.optimizer, loss, var_list, clip_val)

    def minimize(self, sess, feed_dict):
        for _ in range(self.iteration):
            _, grad = sess.run([self.train_op, self.grad], feed_dict)

class ScipyADAMOptimizerInterface(ADAMOptimizerInterface):
    def __init__(self,
                 loss,
                 var_list=None,
                 lr=1e-3,
                 clip_val=10.0,
                 iteration=500,
                 **optimizer_kwargs):
        super(ScipyADAMOptimizerInterface, self).__init__(loss, var_list, lr, clip_val, iteration, **optimizer_kwargs)
        self.scipy_optimizer = ScipyOptimizerInterface(loss, var_list)

    def minimize(self, sess, feed_dict):
        super(ScipyADAMOptimizerInterface, self).minimize(sess, feed_dict)
        self.scipy_optimizer.minimize(sess, feed_dict)



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var.name.replace(':', '_')):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    # tf.summary.scalar('max', tf.reduce_max(var))
    # tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value
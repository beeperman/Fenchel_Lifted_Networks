import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from sklearn.linear_model import Ridge

class RidgeInterface(object):
    """
    Ridge regression optmizer interface
    """

    def __init__(self, W_var, b_var, X, X_next, W_offset=None, sample_weight=None, **kwargs):

        assert isinstance(W_var, tf.Variable)
        assert isinstance(b_var, tf.Variable)
        assert W_offset is None or W_offset.shape == W_var.shape

        self.W_var = W_var
        self.b_var = b_var
        self.W_offset = W_offset

        self.X = X
        self.y = X_next if self.W_offset is None else (X_next - tf.matmul(X, W_offset))
        self.sample_weight = sample_weight

        self.clf = Ridge(**kwargs)

        self._vars = [self.W_var, self.b_var]

        self._update_placeholders = [
            array_ops.placeholder(var.dtype) for var in self._vars
        ]
        
        self._var_updates = [
            var.assign(array_ops.reshape(placeholder, _get_shape_tuple(var)))
            for var, placeholder in zip(self._vars, self._update_placeholders)
        ]


    def minimize(self, session=None, feed_dict=None, fetches=None, **run_kwargs):

        initial_list = [self.X, self.y]

        if self.W_offset is not None:
            initial_list.append(self.W_offset)

        initial_values = session.run(initial_list, feed_dict=feed_dict)

        var_vals = self._minimize(initial_values)

        if self.W_offset is not None:
            var_vals[0] += initial_values[-1]

        session.run(
            self._var_updates,
            feed_dict=dict(zip(self._update_placeholders, var_vals)),
            **run_kwargs)
    
    def _minimize(self, initial_values):
        
        # Replace this method with other optimization functions if needed.
        # Return values must be mappable to the TF Variables in var_list

        minimize_kwargs = {
        'X': initial_values[0],
        'y': initial_values[1],
        'sample_weight': self.sample_weight
        }

        self.clf.fit(**minimize_kwargs)
        W = self.clf.coef_.T
        b = self.clf.intercept_

        assert W.shape == self.W_var.get_shape()
        assert b.shape == self.b_var.get_shape()

        return [W, b]


def _get_shape_tuple(tensor):
  return tuple(dim.value for dim in tensor.get_shape())

import tensorflow as tf
import numpy as np
from tensorflow.contrib.opt import ScipyOptimizerInterface
from nn import activations, lnn_activation_ops, lnn_activation_ops_new, lnn_loss_ops, f_norm_sq, ADAMOptimizerInterface, ScipyADAMOptimizerInterface, variable_summaries
from ridge import RidgeInterface
from closed import XFINInterface
#from utils import variable_summaries
from tensorflow.contrib.layers.python.layers import initializers

class Layer(object):
    """
    The base class for different layers of Lifted Neural Network
    Modified from corey's implementation.
    """

    def __init__(self, input_tensor, num_outputs, num_input_rows, rho=1.0, rhod=1.0, lmbda=1.0, lbd_coef=1.0, activation='relu', loss='cross_entropy', is_first_layer=False, is_last_layer=False, layer_name="layer", pre_fwd=None, post_fwd=None, dtype=tf.float32):
        """
        Initialize some parameters.
        The parameters needed to set
        :param input_tensor:
            A tensor, input
        :param num_outputs:
            An int, equals to shape(output)[-1]
        :param num_input_rows:
            An int, number of data points or batch size
        :param rho:
            A float, coeff for ridge term
        :param rhod:
            A float, coeff for offset term
        :param lmbda:
            A float, coeff for entire layer loss term
        :param lbd_coef:
            A float, the X-objective is computed: lbd_coef * lnn_loss_next + lnn_loss_cur
            this coefficient apply additional attention to loss in the next layer, making the losses
            "back propagate" to the previous layers more quickly.
        :param activation:
            A string, name of the activation function
        :param loss:
            A string, type of the loss if is last layer
        :param is_first_layer: A bool
        :param is_last_layer: A bool
        :param layer_name: A string
        :param pre_fwd:
            A function taking ONE tensor and return a tensor. Applied before the forwarding operation
        :param post_fwd:
            A function taking ONE tensor and return a tensor. Applied after the forwarding operation before activation
        :param dtype: A Tensorflow type
        """
        self.name = layer_name

        self.input_tensor = input_tensor
        self.num_outputs = num_outputs
        self.num_input_rows = num_input_rows

        self.rho  = rho
        self.rhod = rhod
        self.lmbda= lmbda
        self.lbd_coef = lbd_coef
        self.dtype = dtype

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.activation = activation.lower()
        self.loss = loss.lower()

        # the layer pointer should be pointing to a Layer
        # or a tensor of data points or labels (for first and last layer)
        self.prev_layer = None
        self.next_layer = None

        self.X = None
        self.X_shape = None
        self.X_initializer = None
        self.W = None
        self.W_0 = None
        self.W_shape = None
        self.W_initializer = None
        self.b = None
        self.b_0 = None
        self.b_shape = None
        self.b_initializer = None

        # reinitialize the representation variables when batched training
        self.X_reinitialize_op = tf.no_op()
        self.W_0_update_op = tf.no_op()
        self.b_0_update_op = tf.no_op()

        # losses and optimizers for lnn
        self.X_next = None
        self.X_loss = None
        self.Wb_loss = None
        self.Wb_multiplier = 1.0
        self.representation_optimizer = None
        self.weight_optimizer = None
        self.bias_optimizer = None

        self.layer_loss = None

        # the calls to give intermediate outputs between layers
        self.penalty = None
        self.call = None

        # additional operation functions before and after the forwarding_operation
        self.pre_fwd = pre_fwd
        self.post_fwd = post_fwd

        # some flags
        self.tensor_initialized = False
        self.lnn_loss_initialized = False

    def init_shapes_and_initializers(self):
        """
        Initialize the (W, b, X)-shapes and the (W, b, X)-initializers
        self.X_shape
        self.W_shape
        self.b_shape
        self.X_initializer
        self.W_initializer
        self.b_initializer
        :return:
            None
        """
        raise NotImplementedError

    def forwarding_operation(self, input_tensor, weight, bias):
        """
        Forwarding operation without an activation function
        :param input_tensor: A tensor, may be a variable
        :param weight: A tensor, the weight parameters
        :param bias: A tensor, the bias parameters
        :return: A tensor, the output of this layer without an activation function
        """
        raise NotImplementedError

    # TODO: inspect X_next to see what should be needed for calculation of the losses.
    def X_objective(self, X_next):
        """
        Calculates the loss used to update X-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer:
            #loss = lnn_activation_ops[self.activation](X_next, self.penalty) \
            loss = self.lbd_coef * lnn_activation_ops_new[self.loss](X_next, self.penalty) \
                   + self.prev_layer.lmbda * lnn_activation_ops_new[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        else:
            loss = self.lbd_coef * self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty) \
                   + self.prev_layer.lmbda * lnn_activation_ops_new[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        return loss
        raise NotImplementedError

    def Wb_objective(self, X_next):
        """
        Calculates the loss used to update Wb-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer and self.loss == 'cross_entropy':
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.penalty, labels=X_next)) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * (f_norm_sq(self.W))
        else:
            loss = self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * (f_norm_sq(self.W))
        return loss
        raise NotImplementedError

    def X_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        if self.prev_layer.activation == 'relu':
            return ScipyOptimizerInterface(self.X_loss, var_list=[self.X], var_to_bounds={self.X: (0, np.infty)})#, options={'ftol': 2e-15, 'gtol': 1e-11, 'maxls': 100})
        else:
            return ScipyOptimizerInterface(self.X_loss, var_list=[self.X])
        raise NotImplementedError

    def W_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        return ScipyOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b])#, options={'ftol': 2e-15, 'gtol': 1e-15, 'maxls': 100, 'eps': 1e-12})
        raise NotImplementedError

    def b_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        if self.is_last_layer and self.loss == 'cross_entropy':
            #return ScipyOptimizerInterface(self.Wb_loss, var_list=[self.b])
            return None
        else:
            return None
        raise NotImplementedError


    def init(self):
        """
        Call this after setting parameters (especially the parameters used for initializers)
        Initialize the variables **after** initialization of the (W, b, X)-initializers
        :return:
            None
        """
        with tf.variable_scope(self.name):
            self.init_shapes_and_initializers()

            self.X = self.init_representations()
            self.W, self.W_0, self.W_0_update_op = self.init_weights()
            self.b, self.b_0, self.b_0_update_op = self.init_biases()

            # LNN penalty
            self.penalty = self.lnn_penalty()

            # feed forward tensor after activation if any
            self.call = self.feed_forward()

            self.tensor_initialized = True

    def init_fit(self):
        """
        Call this after connecting the layers so that the lifted losses are prepared and ready to be called.
        :return:
            None
        """
        if self.is_last_layer:
            self.X_next = self.next_layer
        else:
            self.X_next = self.next_layer.X
        if not self.is_first_layer:
            # the first layer doesn't need the representation variables
            # also it doesn't possess an X_loss
            self.init_fit_representations()
        self.init_fit_weights()

        self.lnn_loss_initialized = True

    def lnn_penalty(self):
        """
        Calculates WX_l + b1^T, used to construct the loss L = ||WX_l + b1^T - X_l+1||^2 + rho||W||^2
        :return:
            A tensor, the output of the forward pass that takes the representation variables as input.
        """
        return self._forwarding_operation(self.X, self.W, self.b)

    def feed_forward(self):
        """
        Simple feed forward network
        :return:
            A tensor, the output tensor from feed forward network
        """
        call = activations[self.activation](self._forwarding_operation(self.input_tensor, self.W, self.b))
        variable_summaries(call)
        return call

    def _forwarding_operation(self, input_tensor, weight, bias):
        """
        Forwarding wrapper. Add pre_forwarding and post_forwarding operations
        :param input_tensor: A tensor, may be a variable
        :param weight: A tensor, the weight parameters
        :param bias: A tensor, the bias parameters
        :return: A tensor, the output of this layer without an activation function
        """
        if self.pre_fwd is not None:
            input_tensor = self.pre_fwd(input_tensor)

        op = self.forwarding_operation(input_tensor, weight, bias)

        if self.post_fwd is not None:
            op = self.post_fwd(op)

        return op


    def init_weights(self):
        """
        Initialize weights
        :return:
            A tensor, the initialized weight variable
            A tensor, the old weights
        """
        if self.W_initializer is None:
            return None, None, tf.no_op()

        W = tf.get_variable('weight', shape=self.W_shape, initializer=self.W_initializer, dtype=self.dtype)
        W_0 = tf.get_variable('weight_0', initializer=tf.zeros_like(W), dtype=self.dtype)
        update_op = W_0.assign(W)
        variable_summaries(W)
        return W, W_0, update_op

    def init_biases(self):
        """
        Initialize biases
        :return:
            A tensor, the initialized weight variable
        """
        if self.b_initializer is None:
            return None, None, tf.no_op()

        b = tf.get_variable('bias', shape=self.b_shape, initializer=self.b_initializer, dtype=self.dtype)
        b_0 = tf.get_variable('bias_0', initializer=tf.zeros_like(b), dtype=self.dtype)
        update_op = b_0.assign(b)
        variable_summaries(b)
        return b, b_0, update_op

    def init_representations(self):
        """
        Initialize representations of the input to this layer
        except for the first layer where the representations are the data points
        :return:
            A tensor, the initialized representation variable
        """
        if self.X_initializer is None:
            return None

        if self.is_first_layer:
            return self.input_tensor

        X = tf.get_variable('representation', shape=self.X_shape, initializer=self.X_initializer, dtype=self.dtype)
        variable_summaries(X)
        return X

    def init_fit_representations(self):
        """
        Initializes the representation optimization op during BCD as || X_(l+1) - (W_l * X_l + b_l) ||^2_F + || X_l - (W_(l-1) * X_(l-1) + b_(l-1)) ||^2_F where X is your representation.
        Last layer can be represented as a cross entropy loss with some ground truth.
        Optimizes with scipy.optimize.minimize
        :return:
            None
        """
        if self.X_initializer is None:
            return

        with tf.variable_scope("fit_representations"):
            with tf.variable_scope("objective"):
                self.X_loss = self.X_objective(self.X_next)
            tf.summary.scalar("{}_X_loss".format(self.name), self.X_loss)

            # optimizer for the representation variables
            self.representation_optimizer = self.X_optimizer()

            # assign by feed forward
            self.X_reinitialize_op = self.X.assign(self.prev_layer.call)


    def init_fit_weights(self):
        """
        initializes the weight optimization op during BCD as || X_(l+1) - (W_l * X_l + b_l) ||^2_F with an additional ridge regularization term on W_l.
        Last layer can be represented as a cross entropy loss with some ground truth
        Optimizes with an l-bfgs solver
        :return:
            None
        """
        if self.W_initializer is None:
            return

        with tf.variable_scope("fit_weights"):
            with tf.variable_scope("objective"):
                self.Wb_loss = self.Wb_multiplier * self.Wb_objective(self.X_next)
            tf.summary.scalar("{}_Wb_loss".format(self.name), self.Wb_loss)

            # optimizer for the weight variables
            self.weight_optimizer = self.W_optimizer()
            self.bias_optimizer = self.b_optimizer()

    def set_prev_layer(self, prev):
        """
        Set a pointer to the previous layer to use during optimization
        :return:
            None
        """
        self.prev_layer = prev

    def set_next_layer(self, next):
        """
        Set a pointer to the next layer to use during optimization
        :return:
            None
        """
        self.next_layer = next



class FCLayer(Layer):
    """
    The fully connected layer
    """

    def __init__(self, input_tensor, num_outputs, num_input_rows, rho=1.0, rhod=1.0, lmbda=1.0, lbd_coef=1.0, activation='relu', loss='cross_entropy', is_first_layer=False, is_last_layer=False, layer_name='fc', pre_fwd=None, post_fwd=None, dtype=tf.float32):
        """
        Initialize some parameters.
        :param input_tensor:
            A tensor, input
        :param num_outputs:
            An int, equals to shape(output)[-1]
        :param num_input_rows:
            An int, number of data points or batch size
        :param rho:
            A float, coeff for ridge term
        :param rhod:
            A float, coeff for offset term
        :param lmbda:
            A float, coeff for entire layer loss term
        :param lbd_coef:
            A float, the X-objective is computed: lbd_coef * lnn_loss_next + lnn_loss_cur
            this coefficient apply additional attention to loss in the next layer, making the losses
            "back propagate" to the previous layers more quickly.
        :param activation:
            A string, name of the activation function
        :param loss:
            A string, type of the loss if is last layer
        :param is_first_layer: A bool
        :param is_last_layer: A bool
        :param layer_name: A string
        :param pre_fwd:
            A function taking ONE tensor and return a tensor. Applied before the forwarding operation
        :param post_fwd:
            A function taking ONE tensor and return a tensor. Applied after the forwarding operation before activation
        :param dtype: A Tensorflow type
        """
        assert isinstance(num_outputs, int), 'num_outputs must be an int'

        super(FCLayer, self).__init__(input_tensor, num_outputs, num_input_rows, rho=rho, rhod=rhod, lmbda=lmbda, lbd_coef=lbd_coef, activation=activation, loss=loss, is_first_layer=is_first_layer, is_last_layer=is_last_layer, pre_fwd=pre_fwd, post_fwd=post_fwd, layer_name=layer_name, dtype=dtype)

        # Additional parameters needed for FC layers
        self.in_units = self.input_tensor.get_shape().as_list()[-1]
        self.out_units = num_outputs

        self.init()

    def init_shapes_and_initializers(self):
        """
        Initialize the (W, b, X)-shapes and the (W, b, X)-initializers
        self.X_shape
        self.W_shape
        self.b_shape
        self.X_initializer
        self.W_initializer
        self.b_initializer
        :return:
            None
        """
        input_shape = self.input_tensor.get_shape().as_list()
        if len(input_shape) == 4: # pooling will be applied
            self.in_units = np.prod(input_shape[1:]) / 4.0
        self.X_shape = [self.num_input_rows] + input_shape[1:]
        #self.X_shape = [self.num_input_rows, self.in_units  ]
        self.W_shape = [self.in_units, self.out_units       ]
        self.b_shape = [self.out_units                      ]
        self.X_initializer = tf.random_uniform_initializer(maxval=0.1, dtype=self.dtype)
        #self.X_initializer = initializers.xavier_initializer()
        self.W_initializer = initializers.xavier_initializer(dtype=self.dtype)
        #self.b_initializer = initializers.xavier_initializer()
        self.b_initializer = tf.constant_initializer(0.1, dtype=self.dtype)

    def forwarding_operation(self, input_tensor, weight, bias):
        """
        Forwarding operation without an activation function
        :param input_tensor: A tensor, may be a variable
        :param weight: A tensor, the weight parameters
        :param bias: A tensor, the bias parameters
        :return: A tensor, the output of this layer without an activation function
        """
        op = tf.matmul(input_tensor, weight) + bias
        return op

    def X_objective(self, X_next):
        """
        Calculates the loss used to update X-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer:
            #loss = lnn_activation_ops[self.activation](X_next, self.penalty) \
            loss = lnn_activation_ops[self.loss](X_next, self.penalty) \
                   + self.prev_layer.lmbda * lnn_activation_ops[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        else:
            loss = self.lmbda * lnn_activation_ops[self.activation](X_next, self.penalty) \
                   + self.prev_layer.lmbda * lnn_activation_ops[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        return loss
        raise NotImplementedError

    def Wb_objective(self, X_next):
        """
        Calculates the loss used to update Wb-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer:
            #loss = tf.reduce_sum(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.penalty, labels=X_next))
            #      + self.rhod * tf.reduce_sum(tf.square(self.W - self.W_0)) \
            #      + self.rho * (f_norm_sq(self.W) + f_norm_sq(self.b))
            loss = lnn_activation_ops[self.loss](X_next, self.penalty) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * (f_norm_sq(self.W))
        else:
            loss = self.lmbda * lnn_activation_ops[self.activation](X_next, self.penalty) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * (f_norm_sq(self.W))
        return loss

    def W_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        alpha = self.rhod if self.rhod > 0.0 else self.rho
        if self.is_last_layer and self.loss == 'cross_entropy':
            return ScipyOptimizerInterface(self.Wb_loss, var_list=[self.W])
            #return RidgeInterface(self.W, self.b, self.X, self.X_next, W_offset=self.W_0, alpha=alpha, normalize=False)
        elif self.is_last_layer and self.loss == 'none':
            return RidgeInterface(self.W, self.b, self.X, self.X_next, W_offset=self.W_0, alpha=alpha, normalize=False)
        else:
            # optimize both W and b
            #return ScipyOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b])
            return RidgeInterface(self.W, self.b, self.X, self.X_next, W_offset=self.W_0, alpha=alpha/self.lmbda, normalize=False)


class FC2Layer(FCLayer):
    """
    New fully-connected layer that use a new penalty
    """

    def X_objective(self, X_next):
        """
        Calculates the loss used to update X-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer:
            #loss = lnn_activation_ops[self.activation](X_next, self.penalty) \
            loss = self.lbd_coef * lnn_activation_ops_new[self.loss](X_next, self.penalty) \
                   + self.prev_layer.lmbda * lnn_activation_ops_new[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        else:
            loss = self.lbd_coef * self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty) \
                   + self.prev_layer.lmbda * lnn_activation_ops_new[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        return loss

    def Wb_objective(self, X_next):
        """
        Calculates the loss used to update Wb-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer:
            #loss = tf.reduce_sum(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.penalty, labels=X_next)) + self.rhod * tf.reduce_sum(tf.square(self.W - self.W_0))
            loss = lnn_activation_ops_new[self.loss](X_next, self.penalty) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * (f_norm_sq(self.W))
        else:
            loss = self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * (f_norm_sq(self.W))
        return loss

    def W_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        return ScipyOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b])
        #return ADAMOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b], iteration=1, lr=1e-3)

class RobustLayer(FC2Layer):
    """
    A Robust first layer fully connected
    """
    def Wb_objective(self, X_next):
        """
        Calculates the loss used to update Wb-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        if self.is_last_layer:
            #loss = tf.reduce_sum(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.penalty, labels=X_next)) + self.rhod * tf.reduce_sum(tf.square(self.W - self.W_0))
            loss = tf.sqrt(lnn_activation_ops_new[self.loss](X_next, self.penalty)) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * tf.sqrt(f_norm_sq(self.W))
        else:
            loss = tf.sqrt(self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty)) \
                   + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
                   + self.rho * tf.sqrt(f_norm_sq(self.W))
        return loss

class ConvLayer(Layer):
    """
    The convolutional layer
    """
    def __init__(self, input_tensor, num_outputs, kernel_size, stride, num_input_rows, padding='VALID', rho=1.0, rhod=1.0, lmbda=1.0, lbd_coef=1.0, activation='relu', loss='cross_entropy', is_first_layer=False, is_last_layer=False, layer_name='conv', pre_fwd=None, post_fwd=None, dtype=tf.float32, _lr=1e-3):
        """
        Initialize some parameters.
        :param input_tensor:
            A tensor, input
        :param num_outputs:
            An int, equals to shape(output)[-1]
        :param kernel_size:
            An int
        :param stride:
            An int
        :param padding:
            A string, a parameter for conv2d operation
        :param num_input_rows:
            An int, number of data points or batch size
        :param rho:
            A float, coeff for ridge term
        :param rhod:
            A float, coeff for offset term
        :param lmbda:
            A float, coeff for entire layer loss term
        :param lbd_coef:
            A float, the X-objective is computed: lbd_coef * lnn_loss_next + lnn_loss_cur
            this coefficient apply additional attention to loss in the next layer, making the losses
            "back propagate" to the previous layers more quickly.
        :param activation:
            A string, name of the activation function
        :param loss:
            A string, type of the loss if is last layer
        :param is_first_layer: A bool
        :param is_last_layer: A bool
        :param layer_name: A string
        :param pre_fwd:
            A function taking ONE tensor and return a tensor. Applied before the forwarding operation
        :param post_fwd:
            A function taking ONE tensor and return a tensor. Applied after the forwarding operation before activation
        :param dtype: A Tensorflow type
        :param _lr: A float, for ADAMs
        """
        assert isinstance(num_outputs, int), 'num_outputs must be an int'

        super(ConvLayer, self).__init__(input_tensor, num_outputs, num_input_rows, rho=rho, rhod=rhod, lmbda=lmbda, lbd_coef=lbd_coef, activation=activation, loss=loss, is_first_layer=is_first_layer, is_last_layer=is_last_layer, pre_fwd=pre_fwd, post_fwd=post_fwd, layer_name=layer_name, dtype=dtype)

        # Additional parameters needed for Conv layers
        in_units = self.input_tensor.get_shape().as_list()[-1]
        out_units = num_outputs

        self.ksize = [kernel_size, kernel_size, in_units, out_units]
        self.strides = [1, stride, stride, 1]
        self.padding = padding

        self._lr = _lr

        self.init()

    def init_shapes_and_initializers(self):
        """
        Initialize the (W, b, X)-shapes and the (W, b, X)-initializers
        self.X_shape
        self.W_shape
        self.b_shape
        self.X_initializer
        self.W_initializer
        self.b_initializer
        :return:
            None
        """
        input_shape = self.input_tensor.get_shape().as_list()
        self.X_shape = [self.num_input_rows] + input_shape[1:]
        self.W_shape = self.ksize
        self.b_shape = [self.ksize[-1]]
        self.X_initializer = tf.random_uniform_initializer(maxval=0.1, dtype=self.dtype)
        #self.W_initializer = tf.random_uniform_initializer(maxval=0.1, dtype=self.dtype)
        self.W_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=self.dtype)
        self.b_initializer = tf.constant_initializer(0.1, dtype=self.dtype)

    def forwarding_operation(self, input_tensor, weight, bias):
        """
        Forwarding operation without an activation function
        :param input_tensor: A tensor, may be a variable
        :param weight: A tensor, the weight parameters
        :param bias: A tensor, the bias parameters
        :return: A tensor, the output of this layer without an activation function
        """
        return tf.nn.conv2d(input_tensor, weight, strides=self.strides, padding=self.padding) + bias

    def X_objective(self, X_next):
        """
        Calculates the loss used to update X-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        loss = self.lbd_coef * self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty) \
               + self.prev_layer.lmbda * lnn_activation_ops_new[self.prev_layer.activation](self.X, self.prev_layer.penalty)
        return loss

    def Wb_objective(self, X_next):
        """
        Calculates the loss used to update Wb-variables
        :param X_next: A tensor, the representation variables of the next layer
        :return: A tensor, the loss
        """
        loss = self.lmbda * lnn_activation_ops_new[self.activation](X_next, self.penalty) \
               + self.rhod * (f_norm_sq(self.W - self.W_0) + f_norm_sq(self.b - self.b_0)) \
               + self.rho * (f_norm_sq(self.W) + f_norm_sq(self.b))
        return loss
        raise NotImplementedError

    def W_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        return ScipyOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b])
        #return ScipyADAMOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b], iteration=1, lr=1e-3)
        #return ADAMOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b], iteration=1, lr=1e-3)

class ConvLayerOneStepADAM(ConvLayer):
    """
    Use ADAM one-step optimizer to update weights
    """
    def W_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        return ADAMOptimizerInterface(self.Wb_loss, var_list=[self.W, self.b], iteration=1, lr=self._lr)

class FINLayer(Layer):
    """
    Additional final layer. Additional variables with no weights. Output as is. Usually not used.
    """
    def __init__(self, input_tensor, num_outputs, num_input_rows, lmbda=1.0, activation='none', loss='cross_entropy', is_first_layer=False, is_last_layer=True, layer_name='final', dtype=tf.float32):

        super(FINLayer, self).__init__(input_tensor, num_outputs, num_input_rows, lmbda=lmbda, activation=activation, loss=loss, is_first_layer=is_first_layer, is_last_layer=is_last_layer, layer_name=layer_name, dtype=dtype)

        # Additional parameters needed for FC layers
        self.in_units = self.input_tensor.get_shape().as_list()[-1]
        self.out_units = num_outputs
        assert self.in_units == self.out_units, 'input dimension should equals to the output dimension'

        self.init()

    def init_shapes_and_initializers(self):
        """
        Initialize the (W, b, X)-shapes and the (W, b, X)-initializers
        self.X_shape
        self.W_shape
        self.b_shape
        self.X_initializer
        self.W_initializer
        self.b_initializer
        :return:
            None
        """
        self.X_shape = [self.num_input_rows, self.in_units]
        self.X_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=self.dtype)

    def forwarding_operation(self, input_tensor, weight, bias):
        """
        Forwarding operation without an activation function
        :param input_tensor: A tensor, may be a variable
        :param weight: A tensor, the weight parameters
        :param bias: A tensor, the bias parameters
        :return: A tensor, the output of this layer without an activation function
        """
        return input_tensor

    def X_optimizer(self):
        """
        Get the optimizer that has the method 'minimize' to do the optimization.
        :return:
            An ExternalOptimizerInterface
        """
        if self.loss == 'none' and self.prev_layer.activation == 'none':
            return XFINInterface(self)
        if self.prev_layer.activation == 'relu':
            return ScipyOptimizerInterface(self.X_loss, var_list=[self.X], var_to_bounds={self.X: (0, np.infty)})
        else:
            return ScipyOptimizerInterface(self.X_loss, var_list=[self.X])





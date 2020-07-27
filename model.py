import tensorflow as tf
import numpy as np
#import tensorflow.contrib.layers as layers

from layers import FCLayer, FINLayer, FC2Layer, ConvLayer, ConvLayerOneStepADAM, RobustLayer
from nn import f_norm_sq

TRAINABLE_LAYERS = (FCLayer, FINLayer, FC2Layer, ConvLayer, ConvLayerOneStepADAM, RobustLayer)
class Model(object):

    """
    Batched Lifted Neural Network Model. This is a more general implementation of LNN
    """

    def __init__(self, input_tensor, output_tensor, train_batch_size, rho=0.0, rhod=1.0, lmbda=1.0, train_bcd_alternations=4, lbd_coef=1.0, Wb_multiplier=1.0, use_offset=False, use_neo=False, use_osa=False, dtype=tf.float32):
        """
        Initialize some parameters
        :param input_tensor:
            A tensor, the input of the neural network
        :param output_tensor:
            A tensor, the output of the neural network
        :param train_batch_size:
            An int, the batch size of the training batch. It should be the size of the dataset if not in batch mode.
        :param train_bcd_alternations:
            An int, the alternation within one batch
        :param lbd_coef:
            A float, the lambda magnification coefficient for X-updates
        :param Wb_multiplier:
            A float, the multiplier applied before the Wb loss to make it numerically larger.. Doesn't help tho.
        :param use_offset:
            A bool, use offset in regularization, update W_0 and regularize W-W_0 instead of W.
            When this is set to false, rhod = 0.0 regardless of input
        :param use_neo:
            A bool, use neo punishment to enforce the constraint
        :param use_osa:
            A bool, use one-step ADAM optimizer for conv layers
        """

        self.input = input_tensor
        self.output = output_tensor
        self.output_units = self.output.get_shape().as_list()[-1]
        self.batch_size = train_batch_size
        self.rho = float(rho)
        self.rhod = float(rhod) if use_offset else 0.0
        self.lmbda = float(lmbda)
        self.alternation = train_bcd_alternations
        self.use_offset = use_offset
        self.dtype = dtype

        # additional parameters for all layers
        self.lbd_coef = lbd_coef
        self.Wb_multiplier = Wb_multiplier

        self.layers = []

        # temp variables
        self.fin_type = 'cross_entropy'
        self.last_output = self.input

        # some flags
        self.model_completed = False

        # variables, summary to be assigned
        self.loss = 0
        self.train_op = None
        self.train_op_sgd = None
        self.accuracy = tf.no_op()
        # TODO: move summary out of model
        self.merged_summaries = None

        # Use different layer for neo punishments
        self.FCLayer = FC2Layer if use_neo else FCLayer

        # Use different layer for one-step ADAM in Conv layers
        self.ConvLayer = ConvLayerOneStepADAM if use_osa else ConvLayer

    def add_robust_fc_layer(self, num_outputs, activation='none', lmbda=-1.0, rho=-1.0, rhod=-1.0, lbd_coef=-1.0, pre_fwd=None, post_fwd=None):
        """
        Add a ReLU fully-connected layer at the end of the current neural network
        parameters, see the layers

        :return:
            A tensor, the output of this layer
        """
        lmbda = self.lmbda if lmbda < 0.0 else float(lmbda)
        rho = self.rho if rho < 0.0 else float(rho)
        rhod = self.rhod if rhod < 0.0 else float(rhod)
        lbd_coef = self.lbd_coef if lbd_coef < 0.0 else float(lbd_coef)

        num_layers = len(self.layers)
        is_first_layer = num_layers == 0
        layer = RobustLayer(self.last_output, num_outputs, self.batch_size, layer_name="fcl_{}".format(num_layers), lmbda=lmbda, rho=rho, rhod=rhod, is_first_layer=is_first_layer, activation=activation, pre_fwd=pre_fwd, post_fwd=post_fwd, dtype=self.dtype)
        layer.lbd_coef = lbd_coef
        self.layers.append(layer)
        self.last_output = layer.call
        return self.last_output


    def add_fc_layer(self, num_outputs, activation='relu', lmbda=-1.0, rho=-1.0, rhod=-1.0, lbd_coef=-1.0, pre_fwd=None, post_fwd=None):
        """
        Add a ReLU fully-connected layer at the end of the current neural network
        parameters, see the layers

        :return:
            A tensor, the output of this layer
        """
        lmbda = self.lmbda if lmbda < 0.0 else float(lmbda)
        rho = self.rho if rho < 0.0 else float(rho)
        rhod = self.rhod if rhod < 0.0 else float(rhod)
        lbd_coef = self.lbd_coef if lbd_coef < 0.0 else float(lbd_coef)

        num_layers = len(self.layers)
        is_first_layer = num_layers == 0
        layer = self.FCLayer(self.last_output, num_outputs, self.batch_size, layer_name="fcl_{}".format(num_layers), lmbda=lmbda, rho=rho, rhod=rhod, is_first_layer=is_first_layer, activation=activation, pre_fwd=pre_fwd, post_fwd=post_fwd, dtype=self.dtype)
        layer.lbd_coef = lbd_coef
        self.layers.append(layer)
        self.last_output = layer.call
        return self.last_output


    def add_conv_layer(self, num_outputs, kernel_size, stride, padding='VALID', activation='relu', lmbda=-1.0, rho=-1.0, rhod=-1.0, lbd_coef=-1.0, pre_fwd=None, post_fwd=None, _final=False, _lr=1e-3):
        """
        Add a ReLU convolutional layer at the end of the current neural network
        parameters, see the layers

        :return:
            A tensor, the output of this layer
        """
        lmbda = self.lmbda if lmbda < 0.0 else float(lmbda)
        rho = self.rho if rho < 0.0 else float(rho)
        rhod = self.rhod if rhod < 0.0 else float(rhod)
        lbd_coef = self.lbd_coef if lbd_coef < 0.0 else float(lbd_coef)

        num_layers = len(self.layers)
        is_first_layer = num_layers == 0
        layer = self.ConvLayer(self.last_output, num_outputs, kernel_size, stride, self.batch_size, padding=padding, layer_name="conv_{}".format(num_layers), lmbda=lmbda, rho=rho, rhod=rhod, lbd_coef=lbd_coef, is_first_layer=is_first_layer, is_last_layer=_final, activation=activation, pre_fwd=pre_fwd, post_fwd=post_fwd, dtype=self.dtype, _lr=_lr)
        self.layers.append(layer)
        self.last_output = layer.call
        return self.last_output


    POSSIBLE_TYPES = ('cross_entropy', 'none', 'none_ce')
    def add_final_layer(self, type, coef=1.0, additional_variables=False, lmbda=-1.0, rho=-1.0, rhod=-1.0, lbd_coef=-1.0):
        """
        Add the final layer to the neural network. This function must be called exactly once.
        other parameters, see the layers
        :param type:
            A string, the type of loss applied
            Possible types: cross_entropy, none
        :param coef:
            A tensor, the coefficient of the final loss
        :param additional_variables:
            A bool, whether to use additional variables
        :return:
            A tensor, the output of this layer
        """
        lmbda = self.lmbda if lmbda < 0.0 else float(lmbda)
        rho = self.rho if rho < 0.0 else float(rho)
        rhod = self.rhod if rhod < 0.0 else float(rhod)
        lbd_coef = self.lbd_coef if lbd_coef < 0.0 else float(lbd_coef)

        num_layers = len(self.layers)
        is_first_layer = num_layers == 0
        layer = self.FCLayer(self.last_output, self.output_units, self.batch_size, lmbda=lmbda, rho=rho, rhod=rhod, lbd_coef=lbd_coef, layer_name="fcl_{}".format(num_layers), is_first_layer=is_first_layer, is_last_layer=(not additional_variables), activation='none', loss=type, dtype=self.dtype)
        self.layers.append(layer)
        self.last_output = layer.call
        if additional_variables:
            l = FINLayer(layer.call, self.output_units, self.batch_size, layer_name="{}_{}".format(type, num_layers+1), is_first_layer=is_first_layer, is_last_layer=True, activation='none', loss=type)
            self.layers.append(l)

        return self._finalize_layers(type)

        # connect the layers
        for i in range(1, len(self.layers)):
            # except the first layer
            self.layers[i].set_prev_layer(self.layers[i-1])
            self.layers[i-1].set_next_layer(self.layers[i])
        self.layers[0].set_prev_layer(self.input)
        self.layers[-1].set_next_layer(self.output)

        # prepare the lifted losses
        for l in self.layers:
            ## set some parameters
            l.Wb_multiplier = self.Wb_multiplier
            # set up losses
            l.init_fit()

        # some summary
        if type == 'cross_entropy':
            with tf.variable_scope("nn_loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer.call, labels=self.output))
            tf.summary.scalar('nn_loss', self.loss)

            with tf.variable_scope("nn_train"):
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

            with tf.variable_scope("accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, 1), tf.argmax(layer.call, 1)), self.dtype))
            tf.summary.scalar('accuracy', self.accuracy)
        else: #elif type == 'none':
            with tf.variable_scope("nn_loss"):
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(layer.call, self.output), axis=-1))
            tf.summary.scalar('nn_loss', self.loss)

            with tf.variable_scope("nn_train"):
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.fin_type = type

        self.merged_summaries = tf.summary.merge_all()

        self.model_completed = True

        return layer.call

    def _finalize_layers(self, type):
        """
        The function used to finalize the model construction
        :param type:
            A string, the type of loss applied
            Possible types: cross_entropy, none
        :return:
            A tensor, the output of the final layer
        """
        layer = self.layers[-1]

        # connect the layers
        for i in range(1, len(self.layers)):
            # except the first layer
            self.layers[i].set_prev_layer(self.layers[i-1])
            self.layers[i-1].set_next_layer(self.layers[i])
        self.layers[0].set_prev_layer(self.input)
        self.layers[-1].set_next_layer(self.output)

        # prepare the lifted losses
        for l in self.layers:
            ## set some parameters
            l.Wb_multiplier = self.Wb_multiplier
            # set up losses
            l.init_fit()


        # robust comparison add regularization for first layer
        if self.layers[0].rho > 0.0:
            self.loss += self.layers[0].rho * f_norm_sq(self.layers[0].W) / self.batch_size

        # some summary
        if type == 'cross_entropy':
            with tf.variable_scope("nn_loss"):
                self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer.call, labels=self.output))
            tf.summary.scalar('nn_loss', self.loss)

            with tf.variable_scope("nn_train"):
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
                self.train_op_sgd = tf.train.GradientDescentOptimizer(1e-2).minimize(self.loss)

            with tf.variable_scope("accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, 1), tf.argmax(layer.call, 1)), self.dtype))
            tf.summary.scalar('accuracy', self.accuracy)
        else: #elif type == 'none':
            with tf.variable_scope("nn_loss"):
                self.loss += tf.reduce_mean(tf.reduce_sum(tf.squared_difference(layer.call, self.output), axis=-1))
            tf.summary.scalar('nn_loss', self.loss)

            with tf.variable_scope("nn_train"):
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
                self.train_op_sgd = tf.train.GradientDescentOptimizer(1e-2).minimize(self.loss)

        if type == 'none_ce':
            with tf.variable_scope("accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, 1), tf.argmax(layer.call, 1)), self.dtype))
            tf.summary.scalar('accuracy', self.accuracy)

        self.fin_type = type

        self.merged_summaries = tf.summary.merge_all()

        self.model_completed = True

        return layer.call

    # TODO: move logging out of the model
    def fit_lnn(self, sess, feed_dict, n_batch, train_writer=None, reinitialize_representations=-1, X_first=False, layer_alter=False):
        """
        Use lifted updates to fit the model to the data provided, which may be a sampled batch of data.
        :param sess:
            A Tensorflow session, the current session
        :param feed_dict:
            A dictionary, the feed_dict for Tensorflow
        :param n_batch:
            An int, the current batch count
        :param train_writer:
            A Tensorflow writer
        :param reinitialize_representations:
            An int , determine whether to run feed_forward to initialize the representations for this batch of data
            also determine the number of runs of fit_representation before the actual alternative training.
            useful when each batch has different data
        :param X_first:
            A bool, Ture: to optimize X_variables first. False: to optmizize Wb_variables first
        :param layer_alter:
            A bool, to optimize in the order: Wl, Xl, Wl-1, Xl-1, ...
        :return:
            None
        """
        assert self.model_completed, 'Call add_final_layer first!'

        self.reinitialize_representations = reinitialize_representations

        if reinitialize_representations >= 0:
            sess.run([layer.X_reinitialize_op for layer in self.layers], feed_dict=feed_dict)
            for i in range(reinitialize_representations):
                print("Representation pre-train {}:".format(i))
                self.fit_representations_lnn(sess, feed_dict=feed_dict)

        if self.use_offset:
            self.update_weights(sess)

        for alternation in range(self.alternation):
            print("Training alternation {}:".format(alternation))
            self.fit_XWb_lnn(sess, feed_dict=feed_dict, X_first=X_first, is_final=alternation==self.alternation-1, layer_alter=layer_alter)
            if train_writer:
                train_summary = sess.run(self.merged_summaries, feed_dict=feed_dict)
                train_writer.add_summary(train_summary, (n_batch * self.alternation) + alternation)

        train_loss_val, train_acc_val = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        ret = [train_loss_val]
        print("Batch {} Results: ".format(n_batch))
        print("\tTrain loss: {}".format(train_loss_val))
        if not self.fin_type == 'none':
            ret.append(train_acc_val)
            print("\tTrain accuracy: {}".format(train_acc_val))
        return ret

    def update_weights(self, sess):
        """
        update W_0
        :param sess:
            A Tensorflow session, the current session
        :return:
            None
        """
        for layer in self.layers:
            if isinstance(layer, TRAINABLE_LAYERS):
                sess.run([layer.W_0_update_op, layer.b_0_update_op])


    def fit_XWb_lnn(self, sess, feed_dict, X_first=False, is_final=False, layer_alter=False):
        """
        re-order the operations as specified
        """

        #Code for a different order of updates (from last layer to the first, alternating in each layer. Not very helpful)
        if layer_alter:
            for layer in reversed(self.layers):
                if isinstance(layer, TRAINABLE_LAYERS):
                    if layer.weight_optimizer:
                        W_0 = sess.run(layer.W)
                        layer.weight_optimizer.minimize(sess, feed_dict=feed_dict)
                    if layer.bias_optimizer:
                        layer.bias_optimizer.minimize(sess, feed_dict=feed_dict)
                    if layer.weight_optimizer or layer.bias_optimizer:
                        print('Optimized Weights: {}, Wb_loss: {}, Difference: {}, Magnitude: {}'.format(layer.name, sess.run(layer.Wb_loss, feed_dict), np.sum(np.square(sess.run(layer.W) - W_0)), np.sum(np.square(sess.run(layer.W)))))
                    if not layer.is_first_layer:
                        if layer.representation_optimizer:
                            X_0 = sess.run(layer.X)
                            layer.representation_optimizer.minimize(sess, feed_dict=feed_dict)
                            print('Optimized X: {}, X_loss: {}, Difference: {}'.format(layer.name, sess.run(layer.X_loss, feed_dict), np.sum(np.square(sess.run(layer.X) - X_0))))
        else:
            if X_first:
                self.fit_representations_lnn(sess, feed_dict=feed_dict)
                self.fit_weights_lnn(sess, feed_dict=feed_dict)
            else:
                self.fit_weights_lnn(sess, feed_dict=feed_dict)
                # stop if unnecessary
                if self.reinitialize_representations < 0 or not is_final:
                    self.fit_representations_lnn(sess, feed_dict=feed_dict)

    def fit_weights_lnn(self, sess, feed_dict):
        '''
        iterates through all trainable layers and optimizes their weights
        '''
        for layer in self.layers:
            if isinstance(layer, TRAINABLE_LAYERS):
                if layer.weight_optimizer:
                    W_0 = sess.run(layer.W)
                    layer.weight_optimizer.minimize(sess, feed_dict=feed_dict)
                if layer.bias_optimizer:
                    layer.bias_optimizer.minimize(sess, feed_dict=feed_dict)
                if layer.weight_optimizer or layer.bias_optimizer:
                    print('Optimized Weights: {}, Wb_loss: {}, Difference: {}, Magnitude: {}'.format(layer.name, sess.run(layer.Wb_loss, feed_dict), np.sum(np.square(sess.run(layer.W) - W_0)), np.sum(np.square(sess.run(layer.W)))))

    def fit_representations_lnn(self, sess, feed_dict):
        '''
        iterates through all trainable layers except the first and optimizes their reperesentations
        '''
        for layer in reversed(self.layers):
            #Test what will happen if we stop X-update on some layers
            #if isinstance(layer.prev_layer, TRAINABLE_LAYERS) and layer.prev_layer.num_outputs == 50:
            #    sess.run(layer.X_reinitialize_op, feed_dict)
            #    continue
            if isinstance(layer, TRAINABLE_LAYERS):
                if not layer.is_first_layer:
                    if layer.representation_optimizer:
                        X_0 = sess.run(layer.X)
                        layer.representation_optimizer.minimize(sess, feed_dict=feed_dict)
                        print('Optimized X: {}, X_loss: {}, Difference: {}'.format(layer.name,
                                                                              sess.run(layer.X_loss, feed_dict),
                                                                              np.sum(np.square(
                                                                                  sess.run(layer.X) - X_0))))


    def fit_gd(self, sess, feed_dict, n_batch, train_writer=None, sgd=False):
        """
        Use gradient descent to optimize the neural net. As reference.
        :param sess:
            A Tensorflow session, the current session
        :param feed_dict:
            A dictionary, the feed_dict for Tensorflow
        :param n_batch:
            An int, the current batch count
        :param train_writer:
            A Tensorflow writer
        :return:
            An array, the output of neural network calculated from the batch of data
        """
        assert self.model_completed, 'Call add_final_layer first!'
        train_op = self.train_op_sgd if sgd else self.train_op
        sess.run(train_op, feed_dict=feed_dict)
        train_loss_val, train_acc_val, train_summary = sess.run([self.loss, self.accuracy, self.merged_summaries], feed_dict=feed_dict)
        train_writer.add_summary(train_summary, n_batch)
        print("Batch {} Results: ".format(n_batch))
        print("\tTrain loss: {}".format(train_loss_val))
        if not self.fin_type == 'none':
            print("\tTrain accuracy: {}".format(train_acc_val))
        return [train_loss_val, train_acc_val]


    def predict(self, sess, feed_dict, n_batch, train_writer=None, name="Test"):
        """
        Get the output of the nerural net
        :param sess:
            A Tensorflow session, the current session
        :param feed_dict:
            A dictionary, the feed_dict for Tensorflow
        :param n_batch:
            An int, the current batch count
        :param train_writer:
            A Tensorflow writer
        :return:
            An array, the output of neural network calculated from the batch of data
        """
        test_loss_val, test_acc_val = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        if train_writer and False:
            train_summary = sess.run(self.merged_summaries, feed_dict=feed_dict)
            train_writer.add_summary(train_summary, n_batch * self.alternation)

        ret = [test_loss_val]
        print("Batch {} Results: ".format(n_batch))
        print("\t" + name +" loss: {}".format(test_loss_val))
        if not self.fin_type == 'none':
            ret.append(test_acc_val)
            print("\t"+ name + " accuracy: {}".format(test_acc_val))
        return ret

    def check_feed_batch_lnn(self, feed_dict):
        for feed in feed_dict.items():
            assert feed[1].shape[0] == self.batch_size, 'batch size should be as specified'


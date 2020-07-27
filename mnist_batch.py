import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

from model import Model

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    return session

def train_fcl_lnn(data):
    '''
    testing a lifted fully connected model
    '''
    sess = get_session()


    test_cases = 10000
    batch_size = 500
    n_batch = 10000

    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="X")
    y_true = tf.placeholder(dtype=tf.float32, shape=[None], name="y")
    y = tf.one_hot(indices=tf.cast(y_true, tf.int32), depth=10)
    test_feed = {X: data.test.images[:test_cases], y_true: data.test.labels[:test_cases]}

    model = Model(X, y, batch_size, train_bcd_alternations=1, lmbda=1, rho=0.0, rhod=100, use_offset=True, use_neo=True)
    model.add_fc_layer(300, rhod=10)
    model.add_final_layer('cross_entropy', lbd_coef=10)

    train_writer = tf.summary.FileWriter('./logdir/train_lnn_fcl')
    test_writer = tf.summary.FileWriter('./logdir/test_lnn_fcl')

    sess.run(tf.global_variables_initializer())
    for i in range(n_batch):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        train_feed = {X: batch_xs, y_true: batch_ys}
        model.fit_lnn(sess, train_feed, i, train_writer, reinitialize_representations=0, X_first=True)
        model.predict(sess, test_feed, i, test_writer)


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    train_fcl_lnn(input_data.read_data_sets('MNIST_data', one_hot=False))

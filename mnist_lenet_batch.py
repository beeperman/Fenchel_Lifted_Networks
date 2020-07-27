import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from nn import flatten, max_pool, avg_pool

from model import Model

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    return session

def train_cnn_lnn(data):
    '''
    testing a lifted fully connected model
    '''
    sess = get_session()


    test_cases = 10000
    batch_size = 500
    n_batch = 2000

    X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1], name="X")
    y_true = tf.placeholder(dtype=tf.float32, shape=[None], name="y")
    y = tf.one_hot(indices=tf.cast(y_true, tf.int32), depth=10)
    test_feed = {X: np.pad(data.test.images[:test_cases], ((0,0),(2,2),(2,2),(0,0)), 'constant'), y_true: data.test.labels[:test_cases]}

    model = Model(X, y, batch_size, train_bcd_alternations=1, lmbda=1.0, rhod=1.0, use_offset=True, use_neo=True)
    model.add_conv_layer(num_outputs=6, kernel_size=5, stride=1, padding='VALID', rhod=0.0001)
    model.add_conv_layer(num_outputs=16, kernel_size=5, stride=1, padding='VALID', pre_fwd=avg_pool, rhod=0.01, lbd_coef=1.0)
    model.add_fc_layer(120, pre_fwd=lambda x: flatten(avg_pool(x)), lbd_coef=5.0)
    model.add_fc_layer(84, lbd_coef=1.0)
    model.add_final_layer('cross_entropy', lmbda=1.0, rhod=10.0, lbd_coef=5.0)

    train_writer = tf.summary.FileWriter('./logdir/train_lnn_cnn')
    test_writer = tf.summary.FileWriter('./logdir/test_lnn_cnn')

    sess.run(tf.global_variables_initializer())

    for i in range(n_batch):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        batch_xs = np.pad(batch_xs, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        train_feed = {X: batch_xs, y_true: batch_ys}
        model.fit_lnn(sess, train_feed, i, train_writer, reinitialize_representations=0, X_first=True, layer_alter=False)
        model.predict(sess, test_feed, i, test_writer)

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    train_cnn_lnn(input_data.read_data_sets('MNIST_data', one_hot=False, reshape=False))

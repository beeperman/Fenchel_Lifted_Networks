import numpy as np
import scipy.stats
import os
import pickle

class Dataset(object):
    """
    This class is used to load a dataset and provide some universal interfaces
    Construct a subclass for each dataset
    """

    def __init__(self):
        """
        Initialize the dataset
        """
        self.train_data = None
        self.train_data_shape = ()
        self.train_labels = None
        self.train_labels_shape = ()
        self.train_size = 0

        self.test_data = None
        self.test_data_shape = ()
        self.test_labels = None
        self.test_labels_shape = ()
        self.test_size = 0

        self.data_dtype = None

        self.train_data, self.train_labels = self.get_train()
        self.test_data, self.test_labels = self.get_test()

        self.train_data_shape = np.shape(self.train_data)
        self.train_labels_shape = np.shape(self.train_labels)
        self.train_size = self.train_data_shape[0]

        self.test_data_shape = np.shape(self.test_data)
        self.test_labels_shape = np.shape(self.test_labels)
        self.test_size = self.test_data_shape[0]

        self.data_dtype = self.train_data.dtype
        self.labels_dtype = self.train_labels.dtype
        pass

    def get_train(self):
        """
        Get the training data
        :return:
            An array, the first dimension represents data point
        """
        raise NotImplementedError

    def get_test(self):
        """
        Get the test data
        :return:
            An array, the first dimension represents data point
        """
        raise NotImplementedError

    def train(self, batch_size=0, post_op=None):
        """
        Get a training batch of the dataset
        :param batch_size: An int, the batch size of the sampled training batch
        :param post_op: A function, to be called after sampling the batch, usually used to manipulate the shape
        :return:
            An array, the batch
        """
        batch_size = batch_size if batch_size > 0 else self.train_size
        data_batch_shape = [batch_size] + list(self.train_data_shape[1:])
        data_batch = np.zeros(data_batch_shape, self.data_dtype)
        labels_batch_shape = [batch_size] + list(self.train_labels_shape[1:])
        labels_batch = np.zeros(labels_batch_shape, self.labels_dtype)

        # sample tha indices
        batch_indices = np.random.choice(np.arange(self.train_size), size=batch_size, replace=False)

        # populate the batch
        count = 0
        for i in batch_indices:
            data_batch[count] = self.train_data[i]
            labels_batch[count] = self.train_labels[i]
            count += 1

        # apply post_op
        if post_op is not None:
            data_batch, labels_batch = post_op(data_batch, labels_batch)

        return data_batch, labels_batch



    def test(self, post_op=None):
        """
        Get the testing batch of the dataset
        :param post_op: A function, to be called after sampling the batch, usually used to manipulate the shape
        :return:
            An array, the batch
        """
        data_batch = self.test_data
        labels_batch = self.test_labels

        # apply post_op
        if post_op is not None:
            data_batch, labels_batch = post_op(data_batch, labels_batch)

        return data_batch, labels_batch

    @classmethod
    def add_noise(cls, data, noise_func=lambda x, s: scipy.stats.truncnorm.rvs(0, 1, loc=x, scale=0.5, size=s)):
        """
        Add noise to data to create a robust test set
        :param data: A numpy array
        :param noise_func: A function that takes in a size and outputs the noise
        :return: A numpy array
        """
        return data + noise_func(np.array(data), np.shape(data))



class CIFAR10(Dataset):
    """
    The CIFAR-10 dataset
    see https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, data_dir='.', normalize=False):
        """
        Initialize the dataset
        """
        self.data_dir = os.path.abspath(data_dir)

        super(CIFAR10, self).__init__()

        # normalize data if needed
        if normalize:
            mean = [125.307, 122.95, 113.865]
            std = [62.9932, 62.0887, 66.7048]
            for i in range(3):
                self.train_data[..., i] = (self.train_data[..., i] - mean[i]) / std[i]
                self.test_data[..., i] = (self.test_data[..., i] - mean[i]) / std[i]

    def get_train(self):
        """
        Get the training data
        :return:
            data: An array, the first dimension represents data point
            label: An array, the first dimension represents data point
        """
        data_shape = (50000, 32, 32, 3)
        labels_shape = (50000, 10)

        data_dtype = np.float32
        labels_dtype = np.int32

        train_data = np.zeros(data_shape, data_dtype)
        train_labels = np.zeros(labels_shape, labels_dtype)

        for i in range(1, 6):
            with open(os.path.join(self.data_dir, 'data_batch_' + str(i)), 'rb') as f:
                dict = pickle.load(f, encoding='latin1')
            data = dict['data']
            labels = dict['labels']
            data, labels = self.massage(data, labels)
            train_data[(i - 1) * 10000 : i * 10000] = data
            train_labels[(i - 1) * 10000 : i * 10000] = labels

        return train_data, train_labels


    def get_test(self):
        """
        Get the test data
        :return:
            data: An array, the first dimension represents data point
            label: An array, the first dimension represents data point
        """
        data_shape = (10000, 32, 32, 3)
        labels_shape = (10000, 10)

        data_dtype = np.float32
        labels_dtype = np.int32

        test_data = np.zeros(data_shape, data_dtype)
        test_labels = np.zeros(labels_shape, labels_dtype)

        with open(os.path.join(self.data_dir, 'test_batch'), 'rb') as f:
            dict = pickle.load(f, encoding='latin1')
        data = dict['data']
        labels = dict['labels']
        data, labels = self.massage(data, labels)

        test_data[:] = data
        test_labels[:] = labels

        return test_data, test_labels


    def massage(self, data, labels):
        """
        Reshape the data and labels
        :param data: A 10000x3072 numpy array
        :param labels: A list of 10000 numbers
        :return:
            data: A 10000x32x32x3 numpy array
            labels: A 10000x10 numpy array one hot
        """
        data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.eye(10)[np.array(labels).reshape(-1)]
        return data, labels


if __name__ == '__main__':
    # for testing purposes
    data = CIFAR10('./CIFAR10_data')
    sampled_batch_data, sampled_batch_labels = data.train(100)
    print(str(np.shape(sampled_batch_data)))
    print(str(np.shape(sampled_batch_labels)))

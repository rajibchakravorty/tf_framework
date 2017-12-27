
import tensorflow as tf


from data_parser import Parser

device_string = '/device:GPU:0'

epoch_length = 1000
batch_size = 512

learning_rate_info = dict()
learning_rate_info['init_rate'] = 0.01
learning_rate_info['decay_steps'] = 1000
learning_rate_info['decay_factor'] = 0.96
learning_rate_info['staircase']  =True

image_height = 28
image_width  = 28
image_channel  = 1

class_numbers = 10



train_tfrecords = '/opt/ml_data/mnist/mnist_train.tfrecords'
valid_tfrecords = '/opt/ml_data/mnist/mnist_valid.tfrecords'


features={'image':tf.FixedLenFeature([], tf.string),\
    'label': tf.FixedLenFeature([], tf.int64 )}
train_parser = Parser( features, True, image_height, image_width )
valid_parser = Parser( features, False, image_height, image_width )

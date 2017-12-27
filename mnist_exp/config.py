
from os.path import join

import tensorflow as tf


from data_parser import Parser

device_string = '/device:GPU:0'

## definition of epoch in terms of batch number
batch_per_epoch = 5
batch_size = 512

## batches to be used during statistics collections
batch_per_test = 5


learning_rate_info = dict()
learning_rate_info['init_rate'] = 0.01
learning_rate_info['decay_steps'] = 1000
learning_rate_info['decay_factor'] = 0.96
learning_rate_info['staircase']  =True

image_height = 28
image_width  = 28
image_channel  = 1

class_numbers = 10

checkpoint_path = './checkpoints'
model_checkpoint_path = join( checkpoint_path+'model.ckpt')
train_summary_path = join( checkpoint_path, 'train' )
valid_summary_path = join( checkpoint_path, 'valid' )




train_tfrecords = '/opt/ml_data/mnist/mnist_train.tfrecords'
valid_tfrecords = '/opt/ml_data/mnist/mnist_valid.tfrecords'

## information for parsing the tfrecord
features={'image':tf.FixedLenFeature([], tf.string),\
    'label': tf.FixedLenFeature([], tf.int64 )}
train_parser = Parser( features, True, image_height, image_width )
valid_parser = Parser( features, False, image_height, image_width )

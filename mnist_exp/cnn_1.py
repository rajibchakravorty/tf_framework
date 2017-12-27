
import numpy as np
import tensorflow as tf

import tensorflow.contrib as contrib

def cnn_archi( images ):

    with tf.name_scope( 'conv1_1' ) as scope:
        conv1_1 = tf.layers.conv2d( images, filters = 32,
                                    kernel_size = [3,3],
                                    padding = 'same',
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.initializers.truncated_normal(),
                                    kernel_regularizer=contrib.layers.l1_l2_regularizer())


    with tf.name_scope( 'conv1_2' ) as scope:
        conv1_2 = tf.layers.conv2d( conv1_1, filters = 64,
                                    kernel_size = [3,3],
                                    padding = 'same',
                                    activation = tf.nn.relu,
                                    kernel_initializer=tf.initializers.truncated_normal(),
                                    kernel_regularizer=contrib.layers.l1_l2_regularizer())


    # pool1
    pool1 = tf.layers.max_pooling2d( conv1_2, pool_size = (2,2), strides = (2,2) )

    # norm1
    norm1 = tf.layers.batch_normalization(pool1)

    # fc1
    with tf.name_scope('fc1') as scope:

        norm1_flat = tf.layers.flatten( norm1 )
        fc1_out = tf.layers.dense( norm1_flat, units= 512, activation = tf.nn.relu)

    # fc1
    with tf.name_scope('fc2') as scope:
        fc2_out = tf.layers.dense(fc1_out, units=10, activation=tf.nn.relu)

    return fc1_out
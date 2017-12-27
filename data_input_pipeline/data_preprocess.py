
import numpy as np


import tensorflow as tf

'''
tfrecord_list : a list of tfrecord file
parse_function: A function with a signature parse_function( example_proto )
batch_size : an integer denoting the size of the batch

parser function is supplied by client and therefore this
prepare_dataset is independent of the task in hand.

Returns the Dataset
'''


#########################################
# gets an input list of tfrecord file names
# and prepares a datset
#
#########################################


def prepare_dataset( tfrecord_list ,parse_function, batch_size ):

    dataset = tf.data.TFRecordDataset( tfrecord_list )

    dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.map( parse_function )

    dataset = dataset.batch( batch_size )

    return dataset

import numpy as np


import tensorflow as tf

import tensorflow.contrib as contrib

#########################################
# gets an input list of tfrecord file names
# and prepares a datset
# Note : only handles images as data for now
#
#########################################

def prepare_dataset( tfrecord_list ,parse_function, batch_size ):

    dataset = tf.data.TFRecordDataset( tfrecord_list )

    dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.map( parse_function )

    dataset = dataset.batch( batch_size )

    return dataset
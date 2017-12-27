
import random

import numpy as np

import tensorflow as tf

import skimage
from skimage.io import imread

## expects a list
## each element is a tuple, (file_name, label)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def prep_tfrecord( image_file_label, tfrecord_destfile ):

    random.shuffle( image_file_label )

    list_length = len( image_file_label )
    print '{0} files to prepare'.format( list_length )

    writer = tf.python_io.TFRecordWriter(tfrecord_destfile)

    for elem in image_file_label:

        file_name = elem[0]
        label     = elem[1]

        example = tf.train.Example(features=\
            tf.train.Features(feature={\
                'image': _bytes_feature(file_name),
                'label': _int64_feature( [label] ),
            } ) )

        writer.write(example.SerializeToString())

    writer.close()

    print 'TFRecord file save: {0}'.format( tfrecord_destfile )


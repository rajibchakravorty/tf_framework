
import numpy as np

import tensorflow as tf

from data_input_pipeline.data_preprocess import prepare_dataset
from train_step import train_step

class Trainer():

    def __init__(self, network, config):

        self.network     = network
        self.config      = config
        self.train_parser = config.train_parser,
        self.valid_parser = config.valid_parser,
        self.train_tfrecords = config.train_tfrecords
        self.valid_tfrecords = config.valid_tfrecords

    def train(self):

        g = tf.Graph()

        training_dataset = prepare_dataset( self.train_tfrecords,
                                            self.train_parser[0].parse_example,
                                            self.config.batch_size ).repeat()

        validation_dataset = prepare_dataset( self.valid_tfrecords,
                                              self.valid_parser[0].parse_example,
                                              self.config.batch_size )

        with g.as_default():

            sess = tf.Session()

            handle = tf.placeholder(tf.string, shape=[])
            data_iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
            next_element = data_iterator.get_next()


            train_data_iterator = training_dataset.make_one_shot_iterator()
            #valid_data_iterator = validation_dataset.make_initializable_iterator()

            train_handle = sess.run( train_data_iterator.string_handle() )
            #valid_handle = sess.run( valid_data_iterator.string_handle() )

            image_placeholder= tf.placeholder( tf.float32,
                                               shape = (None, self.config.image_height,
                                                        self.config.image_width,
                                                        self.config.image_channel ) )
            label_placeholder = tf.placeholder(tf.int64,
                                               shape=(None, ))

            logits, mean_loss, learning_rate,\
            accuracy, total_loss, train_op = train_step(image_placeholder, label_placeholder,
                                                        self.network,
                                                        self.config.learning_rate_info,
                                                        self.config.device_string,)

            sess.run(tf.global_variables_initializer())

            for _ in range( 10 ):

                for _ in range( self.config.epoch_length ):

                    images, labels = sess.run( next_element ,feed_dict={handle:train_handle})
                    print images.shape, labels.shape
                    _, entropy_loss, acc = sess.run( [train_op,mean_loss,accuracy],
                                                     feed_dict={image_placeholder:images,
                                                                label_placeholder:labels} )

                    print entropy_loss, acc
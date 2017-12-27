

import tensorflow as tf


def summary_op( mean_loss, learning_rate=None,
                total_loss = None,
                accuracy = None,
                loss_collections = tf.GraphKeys.LOSSES,
                summary_collection = 'train_summaries' ):

    with tf.name_scope( summary_collection ) as scope:
        tf.summary.scalar( 'mean_loss', mean_loss,
                           collections=summary_collection )

        if total_loss is not None:
            tf.summary.scalar( 'total_loss_summary', total_loss,
                               collections=summary_collection)

        if accuracy is not None:
            tf.summary.scalar( 'accuracy', accuracy, collections=summary_collection )

        if not 'train' in summary_collection:
            losses = tf.get_collection('loss_collections')

            for l in losses:
                tf.summary.scalar(l.op.name, l, collections=summary_collection )

            tf.summary.scalar('learning_rate', learning_rate,
                              collections=summary_collection )

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

    summary_op = tf.summary.merge_all( key = summary_collection )

    return summary_op


def train_step( images, labels, network,
                learning_rate_info, device_string,
                cpu_device = '/device:CPU:0',
                loss_op = tf.losses.sparse_softmax_cross_entropy,
                loss_collections = tf.GraphKeys.LOSSES,
                optimizer = tf.train.AdamOptimizer() ) :

    ##################################################################
    ##### training steps #############################################
    ##### fairly generic for most common classification tasks ########
    ##################################################################

    global_step_number = tf.train.create_global_step()

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay( learning_rate_info['init_rate'],
                                    global_step_number,
                                    learning_rate_info['decay_steps'],
                                    learning_rate_info['decay_factor'],
                                    learning_rate_info['staircase'])
    with tf.device( device_string ):

        logits = network( images )

    with tf.device( cpu_device ):

        pred_class = tf.argmax( logits, axis = 1,
                                name='pred_class' )

        accuracy = tf.reduce_mean( tf.cast( tf.equal(labels, pred_class), tf.float32 ) ,
                                   name = 'accuracy' )

        #TODO: have to make this configurable
        label_one_hot = tf.one_hot( labels, depth = 10 )
        loss = loss_op( labels = labels, logits = logits)

        mean_loss = tf.reduce_mean( loss, name = 'mean_entropy_loss' )
        tf.add_to_collection( loss_collections, mean_loss )

        total_loss = tf.add_n( tf.get_collection( loss_collections ),
                               name = 'total_loss' )

        grads = optimizer.compute_gradients( total_loss )

        train_op = optimizer.apply_gradients( grads,
                                              global_step = global_step_number )
    ###########################################################################

    return logits, mean_loss, learning_rate, accuracy, total_loss, train_op
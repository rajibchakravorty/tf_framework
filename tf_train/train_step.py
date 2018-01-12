'''
This file contains the typical steps of a classification task
'''

import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.contrib import layers

'''
function to prepare the validation summary op.
mean_loss and accuracy are tensors to be summarized
for later viewing in the tensorboard
'''
def prep_valid_summary_op( mean_loss,accuracy ):

    summary_collection = 'valid_summary'

    with tf.name_scope( summary_collection ) as scope:
        mean_loss_op = tf.summary.scalar( 'mean_loss', mean_loss,
                           collections=summary_collection )
        acc_op = tf.summary.scalar( 'accuracy' ,accuracy, collections=summary_collection )

    all_ops = [mean_loss_op]+[acc_op]
    summary_op = tf.summary.merge( all_ops, collections = summary_collection )

    return summary_op

'''
function to prepare the training summary op.
summarizes mean_loss, learning_rate, total_loss and accuracy
in addition, it stores the variable statistics (histogram
and distribution).
'''
def prep_train_summary_op( mean_loss, total_loss,
                           learning_rate,accuracy ):

    summary_collection = 'train_summary'

    with tf.name_scope( summary_collection ) as scope:
        mean_loss_op = tf.summary.scalar( 'mean_loss', mean_loss,
                           collections=summary_collection )

        acc_op = tf.summary.scalar( 'accuracy', accuracy,
                           collections = summary_collection)

        total_loss_op = tf.summary.scalar( 'total_loss_summary', total_loss,
                               collections=summary_collection)

        learn_rate_op = tf.summary.scalar( 'learning_rate', learning_rate,
                           collections=summary_collection )


        # Add histograms for trainable variables.
        variable_ops = list()
        for var in tf.trainable_variables():
           variable_ops.append( tf.summary.histogram(var.op.name, var,
                                                      collections=summary_collection) )

    all_ops = [mean_loss_op] + [acc_op] + \
              [total_loss_op] + [learn_rate_op]+variable_ops
    summary_op = tf.summary.merge( all_ops, collections=summary_collection )

    return summary_op

'''
training steps of a typical classification task
Provision for supplying loss calculators and optimizers
'''
#TODO: test supplying other loss_op and optimizers

def train_step( images, labels, output_length, network,
                learning_rate_info, device_string,
                loss_op=tf.losses.sparse_softmax_cross_entropy,
                one_hot=False,
                loss_op_kwargs = None,
                loss_collections=tf.GraphKeys.LOSSES,
                optimizer = tf.train.AdamOptimizer,
                optimizer_kwargs = None,
                cpu_device = '/device:CPU:0',
                 ) :

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

    if optimizer_kwargs is None:
        updater = optimizer( learning_rate = learning_rate )
    else:
        updater = optimizer( learning_rate=learning_rate, **optimizer_kwargs )

    with tf.device( device_string ):
        ## get logits
        logits = network( images )

    with tf.device( cpu_device ):

        ## predict classes
        pred_class = tf.argmax( logits, axis = 1,
                                name='pred_class' )

        ## calculate accuracy
        accuracy = tf.reduce_mean( tf.cast( tf.equal(labels, pred_class), tf.float32 ) ,
                                   name = 'accuracy' )

        ## one hot label and calculate loss
        if one_hot == True:
            label_one_hot = tf.one_hot(labels, depth=output_length)
            if loss_op_kwargs is None:
                loss = loss_op(label_one_hot, logits=logits)
            else:
                loss = loss_op(label_one_hot, logits=logits, **loss_op_kwargs )
        else:
            if loss_op_kwargs is None:
                loss = loss_op(labels=labels, logits=logits)
            else:
                loss = loss_op(labels=labels, logits=logits, **loss_op_kwargs)


        mean_loss = tf.reduce_mean(loss)
        ## collect all losses [includes variable regularization if present]
        #TODO: check if really adds the regularization costs
        total_loss = tf.add_n( tf.get_collection(tf.GraphKeys.LOSSES ) )

        ##calculate gradient and apply it
        grads = updater.compute_gradients( total_loss )

        train_op = updater.apply_gradients( grads,
                                              global_step = global_step_number )
    ###########################################################################

    return logits, mean_loss, learning_rate, accuracy, \
           global_step_number,total_loss, train_op
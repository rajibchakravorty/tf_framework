import os
import struct
import numpy as np

import random

from skimage.io import imsave


def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def get_mnist_data( save_path,dataset = "training",
                      data_path = ".", save_data = False):

    data_list = list( read( dataset, data_path ) )

    mnist_data = list()
    for idx ,dt in enumerate( data_list ):
        label, pixels = dt

        name = 'im_{0}_{1}.jpg'.format(idx, label )

        filename = os.path.join( save_path, name )

        tup = (filename, label )

        mnist_data.append( tup )

        if save_data:

            imsave( filename, pixels )

    if dataset == 'training':
        random.shuffle( mnist_data )


        ##divide into train and valid
        data_length = len( mnist_data )

        train_data = mnist_data[0:50000]
        valid_data = mnist_data[-10000:]

        print 'Train/Valid : {0}/{1}'.format( len(train_data),
                                             len(valid_data) )

        if save_data:
            print '{0} MNIST data saved in {1}'.format( len( mnist_data), save_path )

        return train_data, valid_data

    else:
        print 'Test : {0}'.format(len(mnist_data))

        if save_data:
            print '{0} MNIST data saved in {1}'.format(len(mnist_data), save_path)
        return mnist_data

if __name__ == '__main__':

    data_path = '/opt/ml_data/mnist'
    save_path = '/opt/ml_data/mnist/train'

    _,_ = get_mnist_data(save_path, dataset="training",
                   data_path=data_path, save_data=True)

    save_path = '/opt/ml_data/mnist/test'

    _ = get_mnist_data(save_path, dataset="testing",
                  data_path=data_path, save_data=True)

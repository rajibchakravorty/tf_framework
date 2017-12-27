import tensorflow as tf

default_contrast_lower = 0.3
default_contrast_upper = 0.8

class Parser():

    def __init__(self, features, is_training,
                 resize_height,
                 resize_width):
        self.features = features
        self.is_training = is_training
        self.resize_height = resize_height
        self.resize_width = resize_width

    def parse_example( self, example_proto):

        parsed_features = tf.parse_single_example( example_proto, self.features )

        image_file = parsed_features['image']

        image_string = tf.read_file(image_file)
        image = tf.image.decode_image(image_string)

        image = tf.image.convert_image_dtype( image, tf.float32 )

        label = tf.cast( parsed_features['label'], tf.int64 )


        #if for training, randomly translate the image
        # and then resize

        if self.is_training == True:

            #random_rotation =tf.random_uniform( np.asarray([1]).reshape(1,) )* math.pi / 2.
            #image = contrib.image.rotate(image, random_rotation )
            image = tf.image.random_flip_up_down( image )
            image = tf.image.random_contrast( image, default_contrast_lower,
                                              default_contrast_upper )


        image = tf.image.resize_image_with_crop_or_pad( image, self.resize_height,
                                                        self.resize_width )

        #return (image_file, image, height, width, channel, resize_height, resize_width)

        return image, label
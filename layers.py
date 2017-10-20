import tensorflow as tf

from keras.layers import Input, Activation, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.engine.topology import Layer, InputSpec

class ReflectionPadding2D(Layer):
    '''
        Custom layer implementing reflective padding. Subclass of the keras 
        Layer class
    '''
    
    def __init__(self, padding = (1, 1), **kwargs):

        # Define level of padding
        self.padding = tuple(padding)

        # Specifify the ndim, dtype and shape of every input to a layer
        self.input_spec = [InputSpec(ndim = 4)]

        # Initialise parent class
        super().__init__(**kwargs)
        

    def compute_output_shape(self, s):
        '''
            We modify the shape of the input, so we have to provide the shape
            transformation logic such that Keras can do automatic shape
            inference. In this case the height and width of each channel will
            change with 2. Note that None can be provided for dynamic image size
            at test time.

            Arguments:
                s (tensor): A 4-tensor with (batches, height, width, channels)

            Returns: A 4-tensor witht the transformation logic of the layer.
        '''
        width = s[1] + 2 * self.padding[0] if s[1] is not None else None
        height = s[2] + 2 * self.padding[1] if s[2] is not None else None
        
        return (s[0], width, height, s[3])


    def call(self, x, mask = None):
        '''
            Reflection padding logic. Layer does not support masking.

            Arguments:
                x (tensor):     4-dimensional tensor with the input.

                mask (integer): A mask value - Not supported.

        '''

        # Get padding
        w_pad, h_pad = self.padding

        # Pad in wanted dimensions
        return tf.pad(x,
                      [[0, 0],
                      [h_pad, h_pad],
                      [w_pad, w_pad],
                      [0, 0] ],
                      'REFLECT')


def conv_block(inputs, num_filters, kernel_size, strides = (2, 2),
               padding = 'same', activation = True):
    '''
        Simple convolutional block with batchnormalisation and relu
        activation if specified.

        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)

            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.

            kernel_size (tuple):    An integer or tuple/list of 2 integers,
                                    specifying the width and height of the
                                    2D convolution window. Can be a single 
                                    integer to specify the same value for 
                                    all spatial dimensions.

            strides (tuple):        An integer or tuple/list of 2 integers, 
                                    specifying the strides of the 
                                    convolution along the width and height. 
                                    Can be a single integer to specify the 
                                    same value for all spatial dimensions. 
                                    Specifying any stride value != 1 is 
                                    incompatible with specifying any 
                                    dilation_rate value != 1.

            padding (string):       One of 'valid' or 'same' (case-
                                    insensitive).

            activation (boolean):   Wether or not relu activation should be
                                    used.

        Returns: 4-tensor
    '''

    x = Conv2D(filters = num_filters,
               kernel_size = kernel_size,
               strides = strides,
               padding = padding)(inputs)
    x = BatchNormalization()(x)

    return Activation('relu')(x) if activation else x


def res_block(inputs, num_filters = 64):
    '''
        Define residual resnet style block.
        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)
            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.
        Returns: Residual block tensor
    '''

    # Convolutional blocks
    x = conv_block(inputs,
                   num_filters = num_filters,
                   kernel_size = (3, 3),
                   strides = (1, 1))
    x = conv_block(x,
                   num_filters = num_filters,
                   kernel_size = (3, 3),
                   strides = (1, 1),
                   activation = False)

    # Return as residual layer by adding block input and block output
    return add([x, inputs])


def res_crop_block(inputs, num_filters = 64):
    '''
        Define residual resnet style cropping block as per the style of 
        Johnson et. al.

        Argumentation: For style transfer, we found that standard zero-
        padded convolutions resulted in severe artifacts around the borders 
        of the generated image. We therefore remove padding from the 
        convolutions in residual blocks. A 3 × 3 convolution with no 
        padding reduces the size of a feature map by 1 pixel on each side, 
        so in this case the identity connection of the residual block 
        performs a center crop on the input feature map.

        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)

            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.

        Returns: Residual block tensor
    '''

    # Convolutional blocks
    x = conv_block(inputs,
                   num_filters = num_filters,
                   kernel_size = (3, 3),
                   strides = (1, 1),
                   padding = 'valid')
    x = conv_block(x,
                   num_filters = num_filters,
                   kernel_size = (3, 3),
                   strides = (1, 1),
                   padding = 'valid',
                   activation = False)

    # Crop padding
    inputs = Lambda(lambda x: x[:, 2:-2, 2:-2])(inputs)

    # Return as residual layer by adding block input and block output
    return add([x, inputs])


def deconv_block(inputs, num_filters, kernel_size, shape, strides = (2, 2)):
    '''
        Simple deconvolutional block with batchnormalisation and relu
        activation.

        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)

            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.

            kernel_size (tuple):    An integer or tuple/list of 2 integers,
                                    specifying the width and height of the
                                    2D convolution window. Can be a single 
                                    integer to specify the same value for 
                                    all spatial dimensions.

            shape (tuple):          Shape of the input as a tuple

            strides (tuple):        An integer or tuple/list of 2 integers, 
                                    specifying the strides of the 
                                    convolution along the width and height. 
                                    Can be a single integer to specify the 
                                    same value for all spatial dimensions. 
                                    Specifying any stride value != 1 is 
                                    incompatible with specifying any 
                                    dilation_rate value != 1.

        Returns: 4-tensor
    '''

    x = Conv2DTranspose(num_filters,
                        kernel_size = kernel_size,
                        strides = strides,
                        padding = 'same',
                        output_shape = (None, ) + shape)(inputs)

    x = BatchNormalization()(x)

    return Activation('relu')(x)


def up_block(inputs, num_filters, kernel_size):
    '''
        Simple upsample and convolutional block with batchnormalisation and
        relu activation.
        Args:
            inputs (tensor):        A tensor with input layer (Keras 
                                    functional API)
            num_filters (integer):  An integer with the number of filters to
                                    use in the convolutional layers.
            kernel_size (tuple):    An integer or tuple/list of 2 integers,
                                    specifying the width and height of the
                                    2D convolution window. Can be a single 
                                    integer to specify the same value for 
                                    all spatial dimensions.
    '''

    x = UpSampling2D()(inputs)

    x = Conv2D(num_filters,
               kernel_size = kernel_size,
               padding = 'same')(x)

    x = BatchNormalization()(x)

    return Activation('relu')(x)

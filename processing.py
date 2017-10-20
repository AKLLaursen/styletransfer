import numpy as np
import keras.backend as K

def preproc(x):
    '''
        Demean input using Imagenet means from the original VGG paper and
        reverse channels.

        Args:
            x (tensor): 4-dimensional input tensor (an image of 
                        (batches x height x width x channels))

        Returns: The 4-dimensional input tensor with Imagenet mean 
                 subtracted and channels reversed
    '''

    rn_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32)
    
    return (x - rn_mean)[:, :, :, ::-1]


def get_output(model, layer):
    '''
        Function returning a given VGG conv block layer of specified name.

        Args:
            model (tensor):     A model as tensor to extract model layers
                                from.

            layer (tupple):     A tuple consisting of (block_num, layer_num),
                                where block_num is an integer representing the
                                given block number of VGG to use and layer_num
                                is an integer, representing the conv number to
                                use in the given block. 

        Returns: A tensor representing the given layer
    '''

    return model.get_layer('block_' + str(layer[0]) + '_conv_' \
        + str(layer[1])).output


def gram_matrix(x, shift = -1):
    '''
        Batch calculation the gram matrix based on the given input tensor x.
        Args:
            x (tensor):     A 4-tensor of (batch size, img height, img width,
                            channels)
            shift (int):    A scalar shifting the activations before calculating 
                            the Gram matrix in the style of
                            (Novak & Nikulin, 2016)

        Returns: The calculated gram matrix.
    '''

    # Permutes tensor axis (move channels in front)
    x = K.permute_dimensions(x, (0, 3, 1, 2))

    s = K.shape(x)

    # Flatten img dimensions to a single width x height vector
    feat = K.reshape(x, (s[0], s[1], s[2] * s[3]))

    # Shift the activations before calculating  the Gram matrix in the style of
    #(Novak & Nikulin, 2016)
    feat = feat + shift

    # Calculate Gram matrix
    gram = K.batch_dot(feat,
                       K.permute_dimensions(feat, (0, 2, 1))) \
        / K.prod(K.cast(s[1:], K.floatx()))

    return gram


def mean_sqr(diff):
    '''
        Calculate the mean squared error based on a given input difference.

        Args:
            diff (tensor): Two 4-tensors subtracted from each other

        Returns: The MSE
    '''
    # Get tensor dimensions
    dims = list(range(1, K.ndim(diff)))
    
    # Calculate MSE
    return K.expand_dims(K.sqrt(K.mean(diff ** 2, dims)), 0)


def total_variation_loss(x):

    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])

    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    
    return K.sum(K.pow(a + b, 1.25))
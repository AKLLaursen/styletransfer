import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras_tqdm import TQDMNotebookCallback

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D
import keras.backend as K

from .layers import ReflectionPadding2D, conv_block, res_crop_block, up_block
from .processing import (preproc, get_output, gram_matrix, mean_sqr,
    total_variation_loss)
from .plotting import display_image_in_actual_size

from kerastools.vgg import Vgg

class StyleTransfer(object):

    def __init__(self, shape = (288, 288, 3), style_path = '',
                 w = [0.025, 0.8, 0.15, 0.025], style_weight = 1,
                 content_weight = 1, tv_weight = 1.5e-7,
                 vgg_layers = [(2, 1), (3, 1), (4, 2), (5, 2)]):

        self.shape = shape
        self.style_path = style_path
        self.style = self.read_style()
        self.w = w
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.vgg_layers = vgg_layers

        self.create()


    def read_style(self):
        '''
            Open style image and resize to fit train image dimensions.
        '''

        # Open image using PIL
        style = Image.open(self.style_path)

        # Resize image after height of width depending on which is smaller
        if style.height <= style.width:
            img_size = style.height
        else:
            img_size = style.width

        # Calculate resize factor
        resize_factor = np.divide(style.size,
                                  np.floor(img_size / self.shape[0] \
                                    * 10) / 10.).astype('int32')

        # Resize image based on given dimension
        style = style.resize(resize_factor)

        # Finally make image match inputs
        style = np.array(style)[:self.shape[0], :self.shape[1], :self.shape[2]]

        return style


    def plot_style(self):
        '''
            Plot formatted style image.
        '''

        plt.imshow(self.style)


    def tot_loss(self, x):
        '''
            Calculate the total loss between content loss x and the style loss 
            based on given weights to the perceptual loss (individual conv layer 
            loss).

            Args:
                x (tensor):             A 4-tensor of (batch size, img height, 
                                        img width, channels)

                style_targets (list):   List of style target variables

                w (list):               A list of weights corresponding to the 
                                        number of conv layers used in the 
                                        perceptual loss.

            Returns: The calculated total loss.
        '''

        # Initialise
        loss = 0
        n = len(self.style_targets)

        # Loop over the number of style target (number of conv layers used in
        # perceptual loss).
        for i in range(n):

            # Style loss - The super resolution vgg layers shoould be equal to
            # the style targets.
            loss += mean_sqr(gram_matrix(x[i + n]) \
                - gram_matrix(self.style_targets[i])) * self.style_weight

            # Content loss - The super resolution vgg layers and the non super
            # resolution layers should be equal.
            loss += mean_sqr(x[i] - x[i + n]) * self.w[i] * self.content_weight


        # Total variation loss
        loss += total_variation_loss(x[n * 2]) * self.tv_weight
        return loss


    def create(self):
        '''
            Create style transfer model in the style of Johnston et. al. with 
            given inputs

            Args:

            Returns: The given style transfer model
        '''


        # Define super resolution model
        # Input. Dimensions are set as None in order to use dynamic image size
        inputs = Input((None, None, 3))

        # Reflective layer in order to scale input image up to style size.
        reflective_layer = ReflectionPadding2D((40, 40))(inputs)

        # Convolutional block layers
        conv_layer = conv_block(reflective_layer, 32, (9, 9), (1, 1))
        conv_layer = conv_block(conv_layer, 64, (3, 3))
        conv_layer = conv_block(conv_layer, 128, (3, 3))

        # Residual resnet style blocks
        res_layer = conv_layer
        for i in range(5):
            res_layer = res_crop_block(res_layer, 128)

        # Upsample blocks
        up_layer = up_block(res_layer, 64, (3, 3))
        up_layer = up_block(up_layer, 32, (3, 3))

        # Final super resolution (3 x 288 x 288) conv layer with tanh activation
        final_conv = Conv2D(3,
                            kernel_size = (9, 9),
                            activation = 'tanh',
                            padding = 'same')(up_layer)

        # Transform final tanh activation layer to output signal in range of
        # [0, 255]
        outputs = Lambda(lambda x: (x + 1) * 127.5)(final_conv)

        # Define VGG model (no top) to be used as content model
        vgg_inp = Input(self.shape)
        vgg = Vgg(include_top = False,
                  input_tensor = Lambda(preproc)(vgg_inp),
                  pooling = 'max').model

        # Make the the VGG layers untrainable
        for l in vgg.layers:
            l.trainable = False

        # Define content model, with output from the 4 last content layers
        vgg_content = Model(vgg_inp,
                            [get_output(vgg, layer) for layer in self. vgg_layers])


        # Define the style targets for each content layer as a TF variable. It
        # will be used in the perceptual loss function (simply content loss in)
        # separate conv layers.
        self.style_targets = [K.variable(o) for o in
                             vgg_content.predict(np.expand_dims(self.style, 0))]

        # In Keras' functional API, any layer (and a model is a layer), can be 
        # treated as if it was a function. So we can take any model and pass it 
        # a tensor, and Keras will join the model together.

        # Just the VGG model with perceptual loss.
        vgg1 = vgg_content(vgg_inp)

        # The VGG model with perceptual los on top and the super resolution
        # model on the buttom.
        vgg2 = vgg_content(outputs)

        # Define total loss
        loss = Lambda(self.tot_loss)(vgg1 + vgg2 + [outputs])

        # Define final style transfer model
        self.m_style = Model([inputs, vgg_inp], loss)
        self.inputs = inputs
        self.outputs = outputs


    def compile(self, optimizer = 'adam', loss = 'mae'):
        '''
            Compile model.

            Args:
                None

            Returns:
                Nothing
        '''
        self.m_style.compile(optimizer = optimizer,
                             loss = loss)


    def fit(self, train_data, batch_size, epochs, learning_rate = 1e-3):
        '''
            Function to fit the style transfer model.

            Args:
                train_data (ndarray):   Array of symmetrical train images in 
                                        (288 x 288) size.

                batch_size (int):       Batch size for training

                epochs (int):           Number of epochs to run

            Returns: None

        '''

        # Dictionary to use TQDMN to visualise training
        params = {'verbose': 0,
                 'callbacks': [TQDMNotebookCallback(leave_inner = True)]}

        # Our target is 0 loss of the MSE (final model output)
        target = np.zeros((train_data.shape[0], 1))

        # Set learning rate
        K.set_value(self.m_style.optimizer.lr, learning_rate)

        # Fit model
        self.m_style.fit(x = [train_data, train_data],
                         y = target,
                         batch_size = batch_size,
                         epochs = epochs,
                         **params)


    def get_top(self):
        '''
            Get top layer model for doing style transfer
            
            Args:
                None

            Returns: None
        '''

        self.top_model = Model(self.inputs, self.outputs)


    def plot_image(self, image_path):

        # Read image
        input_image = Image.open(image_path)

        # Convert to numpy array
        input_image = np.expand_dims(np.array(input_image), 0)

        # Plot image
        display_image_in_actual_size(input_image)


    def predict(self, image_path, plot_image = True):

        # Read image
        input_image = Image.open(image_path)

        # Convert to numpy array
        input_image = np.expand_dims(np.array(input_image), 0)

        # Predict (do style transfer)
        self.image_with_style = self.top_model.predict(input_image)

        if plot_image: display_image_in_actual_size(self.image_with_style)
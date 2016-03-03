import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv

class CNN(object):

    def __init__(self, input_shape= (1,96,96),number_of_feature_maps_1,number_of_feature_maps_2,filter_shape)

        """
        input: a 4D tensor corresponding to a mini-batch of input images. The shape of the tensor is as follows:
        [mini-batch size, number of input feature maps, image height, image width].
        """
        self.input = T.tensor4(name='input')

        #Weights
        W_shape = (number_of_feature_maps_2,number_of_feature_maps_1,filter.shape[0],filter.shape[1])
        w_bound = np.sqrt(number_of_feature_maps_1*filter.shape[0]*filter.shape[1])
        self.W =  theano.shared( np.asarray(np.random.uniform(-1.0/w_bound,1,0/w_bound,W_shape),dtype=self.input.dtype), name = 'W' )

        #Bias

        bias_shape = (number_of_feature_maps_2,)
        self.bias = theano.shared(np.asarray(
            np.random.uniform(-.5,.5, size=bias_shape),
            dtype=input.dtype), name ='b')

        #Colvolution

        self.convolution = conv.conv2d(self.input,self.W)
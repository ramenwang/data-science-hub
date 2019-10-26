import tensorflow as tf
import numpy as np 
from functools import partial
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# set up GPU as the first GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# define regularization term
def l2_reg(coef=1e-2):
    return lambda x: tf.reduce_sum(x ** 2) * coef


def l1_reg(coef=1e-2):
    return lambda x: tf.reduce_sum(tf.abs(x)) * coef


# define convolutional layer as a compiled function
@tf.function
def conv_layer(x, kernels, bias, s):
    z = tf.nn.conv2d(input=x, filters=kernels, strides=[1,s,s,1], padding='SAME')
    return tf.nn.relu(z + bias)


class convLayer2d(tf.keras.layers.Layer):
    '''
    Simple convolutional neural networks
    '''

    def __init__(self, num_kernel = 32, kernel_size = (3,3), stride = 1):
        super().__init__()
        self.num_kernel  = num_kernel
        self.kernel_size = kernel_size
        self.stride      = stride

    def build(self, input_shape):
        # assuming shape format BHWC
        num_input_ch  = input_shape[-1]
        kernels_shape = (*self.kernel_size, num_input_ch, self.num_kernel)
        glorot_init   = tf.initializers.GlorotUniform()
        self.kernels  = self.add_weight(name='kernels', shape=kernels_shape, initializer=glorot_init, trainable=True)
        self.bias     = self.add_weight(name='bias', shape=(self.num_kernel,), initializer='random_normal', trainable=True)

    def call(self, input):
        return conv_layer(input, self.kernels, self.bias, self.stride)



class ConvReg2D(convLayer2d):
    '''
    a convolution layer calls 2d convolutional layer function and add regularizer
    '''

    def __init__(self, num_kernel = 32, kernel_size = (3,3), stride = 1,\
        kernel_regularizer = l2_reg(), bias_regularizer = None):
        '''
        initial convLayer2d class and regularization function
        '''
        super().__init__(num_kernel, kernel_size, stride)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shape):
        '''
        call build in convLayer2d and add regularizer to the loss of both kernel and bias
        '''
        super().build(input_shape)
        if self.kernel_regularizer is not None:
            self.add_loss(partial(self.kernel_regularizer, self.kernels))
        if self.bias_regularizer is not None:
            self.add_loss(partial(self.bias_regularizer, self.bias))



if __name__ == "__main__":

    model = Sequential()
    model.add(ConvReg2D(num_kernel=6, kernel_size=(5,5), stride=1, kernel_regularizer=l2_reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ConvReg2D(num_kernel=16, kernel_size=(5,5), stride=1, kernel_regularizer=l2_reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # initializing the trainable variables

    # kernels = tf.Variable(glorot_uni_initializer(kernels_shape), trainable=True, name='filters')
    # bias    = tf.Variable(tf.zeros([N]), trainable=True, name='bias')

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', \
        metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    ]

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    x_train, x_test = x_train[...,tf.newaxis], x_test[...,tf.newaxis]
    model.fit(x_train, y_train, batch_size=24, epochs=80,\
        validation_data=(x_test, y_test), callbacks = callbacks)


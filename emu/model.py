# ------------------------- #
# Creates model for emulator
# ------------------------- #

import tensorflow as tf
# keras imports for building our neural network
from keras.models import Sequential
# keras imports for layers we will put in our model
from keras.layers import Dense, Activation
# keras import for optimizer
from keras.optimizers import Adam


def build_model(n_in, n_out, n_nodes=128, learning_rate=5e-5):    

    activation_type = 'relu'
    use_bias = True

    # build the model
    model = Sequential()    

    model.add(Dense(n_nodes, input_shape=(n_in,), use_bias=use_bias))  
    model.add(Activation(activation_type))

    model.add(Dense(n_nodes, use_bias=use_bias))
    model.add(Activation(activation_type))

    model.add(Dense(n_nodes, use_bias=use_bias))
    model.add(Activation(activation_type))

    model.add(Dense(n_out, use_bias=use_bias))
    model.add(Activation('linear'))

    # compiling the sequential model
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['mse'])  

    # print a helpful summary of our model
    model.summary()

    return model


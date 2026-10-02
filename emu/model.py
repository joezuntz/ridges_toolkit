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


def diagonal_chi2_loss(n_theta):
    def loss(y_true_with_sigma, y_pred):
        # y contains both signal and shape noise, so we first split them
        y_true = y_true_with_sigma[:, :n_theta]
        y_sigma = y_true_with_sigma[:, n_theta:]

        # avoid uncertainty to become too small
        y_sigma = tf.maximum(y_sigma, 1e-8)
        # standardize
        residual_in_sigma = (y_pred - y_true) / y_sigma

        return tf.reduce_mean(tf.square(residual_in_sigma), axis=-1)

    return loss


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
    model.compile(loss=diagonal_chi2_loss(n_out),
                  optimizer=Adam(learning_rate=learning_rate),
                  # metrics=['mse']
                  )    
     
    # print a helpful summary of our model
    model.summary()

    return model


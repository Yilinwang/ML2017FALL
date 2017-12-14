from keras.models import Model
from keras.layers import *
import numpy as np



def basic_256():
    user_input = Input(shape = (1,))
    user_vec = Flatten()(Embedding(6041 + 1, 256, input_length = 1)(user_input))

    movie_input = Input(shape = (1,))
    movie_vec = Flatten()(Embedding(3953 + 1, 256, input_length = 1)(movie_input))

    output = merge([user_vec, movie_vec], mode = 'dot')
    model = Model([user_input, movie_input], output)
    return model


def basic():
    user_input = Input(shape = (1,))
    user_vec = Flatten()(Embedding(6041 + 1, 32, input_length = 1)(user_input))

    movie_input = Input(shape = (1,))
    movie_vec = Flatten()(Embedding(3953 + 1, 32, input_length = 1)(movie_input))

    output = merge([user_vec, movie_vec], mode = 'dot')
    model = Model([user_input, movie_input], output)
    return model


def loss(y, y_pred):
    return (y - y_pred) ** 2 if y != 0 else 0


def get_model(model):
    np.random.seed(0)
    model = globals()[model]()
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model

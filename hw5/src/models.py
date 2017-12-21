from keras.models import Model
from keras.layers import *
import numpy as np



def userf(dim):
    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))
    user_feature_input = Input(shape = (4,))

    user_vec = Flatten()(Embedding(6041, dim, embeddings_initializer = 'random_normal')(user_input))
    user_bias = Flatten()(Embedding(6041, 1, embeddings_initializer = 'zeros')(user_input))
    movie_vec = Flatten()(Embedding(3953, dim, embeddings_initializer = 'random_normal')(movie_input))
    movie_bias = Flatten()(Embedding(3952, 1, embeddings_initializer = 'zeros')(movie_input))

    r_hat = Dot(axes = 1)([user_vec, movie_vec])
    r_hat = Add()([r_hat, user_bias, movie_bias])
    r_hat = Concatenate()([r_hat, user_feature_input])

    nn = Dense(32)(r_hat)
    nn = Dense(1)(nn)

    model = Model([user_input, movie_input, user_feature_input], nn)
    return model


def nobias(dim):
    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))

    user_vec = Flatten()(Embedding(6041, dim, embeddings_initializer = 'random_normal')(user_input))
    movie_vec = Flatten()(Embedding(3953, dim, embeddings_initializer = 'random_normal')(movie_input))

    r_hat = Dot(axes = 1)([user_vec, movie_vec])

    model = Model([user_input, movie_input], r_hat)
    return model


def ta(dim):
    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))

    user_vec = Flatten()(Embedding(6041, dim, embeddings_initializer = 'random_normal')(user_input))
    user_bias = Flatten()(Embedding(6041, 1, embeddings_initializer = 'zeros')(user_input))
    movie_vec = Flatten()(Embedding(3953, dim, embeddings_initializer = 'random_normal')(movie_input))
    movie_bias = Flatten()(Embedding(3952, 1, embeddings_initializer = 'zeros')(movie_input))

    r_hat = Dot(axes = 1)([user_vec, movie_vec])
    r_hat = Add()([r_hat, user_bias, movie_bias])

    model = Model([user_input, movie_input], r_hat)
    return model


def basic_64d03():
    user_input = Input(shape = (1,))
    user_vec = Flatten()(Embedding(6041 + 1, 32, input_length = 1)(user_input))
    user_vec = Dropout(0.3)(user_vec)

    movie_input = Input(shape = (1,))
    movie_vec = Flatten()(Embedding(3953 + 1, 32, input_length = 1)(movie_input))
    movie_vec = Dropout(0.3)(movie_vec)

    output = merge([user_vec, movie_vec], mode = 'dot')
    model = Model([user_input, movie_input], output)
    return model


def bias():
    user_input = Input(shape = (1,))
    user_vec = Flatten()(Embedding(6041 + 1, 32, input_length = 1)(user_input))

    movie_input = Input(shape = (1,))
    movie_vec = Flatten()(Embedding(3953 + 1, 32, input_length = 1)(movie_input))

    output = merge([user_vec, movie_vec], mode = 'dot')
    model = Model([user_input, movie_input], output)
    return model


def basic_64():
    user_input = Input(shape = (1,))
    user_vec = Flatten()(Embedding(6041 + 1, 32, input_length = 1)(user_input))

    movie_input = Input(shape = (1,))
    movie_vec = Flatten()(Embedding(3953 + 1, 32, input_length = 1)(movie_input))

    output = merge([user_vec, movie_vec], mode = 'dot')
    model = Model([user_input, movie_input], output)
    return model


def basic_256():
    user_input = Input(shape = (1,))
    user_vec = Flatten()(Embedding(6041 + 1, 256, input_length = 1)(user_input))
    user_vec = Dropout(0.5)(user_vec)

    movie_input = Input(shape = (1,))
    movie_vec = Flatten()(Embedding(3953 + 1, 256, input_length = 1)(movie_input))
    movie_vec = Dropout(0.5)(movie_vec)

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


def get_model(model, dim):
    np.random.seed(0)
    model = globals()[model](dim)
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model

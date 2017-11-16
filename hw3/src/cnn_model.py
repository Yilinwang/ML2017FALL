from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, normalization, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l2, l1
import numpy as np



def cca4ff():
    model = Sequential()

    lamb = 0

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def vgg_s():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(output_dim=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def vgg():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(output_dim=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def larger():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(output_dim=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def test():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(output_dim=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def dac4_512():
    model = Sequential()

    lamb = 0

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=512, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=512, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def dac4leaky():
    model = Sequential()

    lamb = 0.0000001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def ccaccaccaccaff_da():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def daug_4cm():
    model = Sequential()

    lamb = 0.00001

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def ta_more_m():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='valid', activation='relu', input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(MaxPooling2D((5,5), strides=(2,2)))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model


def ta_more():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='valid', activation='relu', input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(MaxPooling2D((5,5), strides=(2,2)))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model


def ta():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='valid', activation='relu', input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(MaxPooling2D((5,5), strides=(2,2)))
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model
    


def caccaccaccaff_nr():
    model = Sequential()

    lamb = 0

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def ccmccmccmccmff_n2large():
    model = Sequential()

    lamb = 0.001

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def ccmccaccaccaff_n2():
    model = Sequential()

    lamb = 0.001

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def ccmccaccaccaff_n3():
    model = Sequential()

    lamb = 0.0001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def copycat():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='valid', input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid',))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid',))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid',))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid',))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid',))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid',))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    '''
    model.add(Conv2D(512, (3, 3), padding='valid',))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='valid',))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='valid',))
    model.add(ZeroPadding2D())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))
    '''

    model.add(Flatten())
    model.add(Dense(output_dim=1024))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def caccaccaccaff_large():
    model = Sequential()

    lamb = 0.0001

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def caccaccalclcaff_l():
    model = Sequential()

    lamb = 0.0001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def caccaccaccaff_l():
    model = Sequential()

    lamb = 0.00001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(0.05))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def caccaccaccacaff_leaky():
    model = Sequential()

    lamb = 0.0001

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def caccaccaccaff_leaky():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def caccaccaccaff():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(AveragePooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def cmccmccmccmff_zpl2l():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def cmccmccmccmff_zpl2():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l2(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l2(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l2(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l2(lamb)))
    model.add(Activation('softmax'))

    return model


def cmccmccmccmff_zp():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (3, 3), padding='valid', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l1(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l1(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l1(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l1(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l1(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=l1(lamb)))
    model.add(ZeroPadding2D())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l1(lamb)))
    model.add(Activation('softmax'))

    return model


def cmccmccmccmff():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7, kernel_regularizer=l1(lamb)))
    model.add(Activation('softmax'))

    return model


def ccccff_reg6():
    model = Sequential()

    lamb = 0.000001

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model



def cpcpf():
    model = Sequential()

    lamb = 0
    model.add(Conv2D(32, (5, 5), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (7, 7), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(output_dim=256, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7, kernel_regularizer=l1(lamb)))
    model.add(Activation('softmax'))

    return model


def ccpnccpnccpccff():
    model = Sequential()

    lamb = 0.0001
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpncpncpcff_2():
    model = Sequential()

    lamb = 0.0001
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpncpncpcff_im():
    model = Sequential()

    lamb = 0.0001
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpncpncpcff_reg4():
    model = Sequential()

    lamb = 0.0001
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpncpncpcff_reg():
    model = Sequential()

    lamb = 0.001
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(lamb), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=1024, kernel_regularizer=l1(lamb)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpcpcpff():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(output_dim=1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpncpncpcff_lrn():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(normalization.LRN2D())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(normalization.LRN2D())
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(output_dim=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cpncpncpcff():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def ccmccmccmccm_nonorm():
    model = Sequential()
    
    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(72, 5, 5))
    model.add(Conv2D(72, 5, 5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def ccmccmccm():
    model = Sequential()
    
    model.add(Conv2D(32, 3, 3, input_shape=(48, 48, 1)))
    model.add(Conv2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, 3, 3))
    model.add(Conv2D(64, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 3, 3))
    model.add(Conv2D(128, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def ccmccm():
    model = Sequential()
    
    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(72, 5, 5))
    model.add(Conv2D(72, 5, 5))
    model.add(MaxPooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_10_d2():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_6_leakyrelu():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(output_dim=64))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(output_dim=32))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(output_dim=32))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(output_dim=16))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(output_dim=16))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_6_softmax():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=64))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=32))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=32))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=16))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=16))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_6_lastrelu():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('relu'))

    return model


def cmcm_6_lastsigmoid():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('sigmoid'))

    return model


def cmcm_6_lastnoact():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))

    return model


def cmcm_6_fnoact():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_6():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_8():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def cmcm_4up():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cmcm_8():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def ccmcm_8():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=64))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def m_2_4():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def m_3_8():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(78, 5, 5))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def m_2_6():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('sigmoid'))

    return model


def m_depth_2_4():
    model = Sequential()

    model.add(Conv2D(36, 3, 3, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(72, 3, 3))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('sigmoid'))

    return model


def m_depth_2_5():
    model = Sequential()

    model.add(Conv2D(36, 3, 3, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(72, 3, 3))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def m_depth_3_5():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))

    return model


def basic_model():
    model = Sequential()

    model.add(Conv2D(36, 5, 5, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((3,3)))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.add(Activation('softmax'))
    
    return model


def get_model(model, weight=''):
    np.random.seed(0)
    model = globals()[model]()

    if weight != '':
        model.load_weights(weight)
    else:
        model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

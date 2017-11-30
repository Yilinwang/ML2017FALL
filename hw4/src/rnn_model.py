from keras.models import Sequential
from keras.layers import Dense, GRU, RNN, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, Flatten
import numpy as np



def ggggff_sig():
    model = Sequential()

    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(41,100,)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.3))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model


def cpcpff_ave():
    model = Sequential()

    model.add(Conv1D(256, 3, input_shape=(41,100,)))
    model.add(Dropout(0.3))
    model.add(AveragePooling1D())
    model.add(Conv1D(256, 3))
    model.add(Dropout(0.3))
    model.add(AveragePooling1D())

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def cpcpggff():
    model = Sequential()

    model.add(Conv1D(256, 3, input_shape=(41,100,)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(256, 3))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))

    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.3))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def ggff_sig():
    model = Sequential()

    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(41,100,)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model


def ggff():
    model = Sequential()

    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(41,100,)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def bidirect():
    model = Sequential()

    model.add(Bidirectional(GRU(128), input_shape=(41,100,)))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def basic_model():
    model = Sequential()

    model.add(GRU(128, input_shape=(41,100,)))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def get_model(model):
    np.random.seed(0)
    model = globals()[model]()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

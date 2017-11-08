from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten



def m_():
    model = Sequential()

    return model


def basic_model():
    model = Sequential()

    model.add(Conv2D(45, 3, 3, input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(90, 3, 3))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.add(Dense(output_dim=16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=7))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import data_processor, rnn_model
import numpy as np



def train(args):
    X, Y = getattr(data_processor, 'read_train_data')(args.train_path)
    model = rnn_model.get_model(args.model)

    filepath = 'model/' + args.prefix + args.model + '_{epoch:03d}_{val_acc:.2f}'
    checkpoint = ModelCheckpoint(filepath)
    earlyStopping = EarlyStopping(monitor='loss', patience=3)
    callbacks_list = [checkpoint, earlyStopping]

    history = model.fit(X, Y, validation_split = 0.1, epochs = 100, batch_size = 32, callbacks = callbacks_list)


def infer(args):
    X = getattr(data_processor, 'read_test_data')(args.test_path)
    model = load_model(args.weight)

    with open(args.output, 'w') as fp:
        fp.write('id,label\n')
        for idx, p in enumerate(model.predict(X)):
            fp.write(str(idx) + ',' + str(np.argmax(p)) + '\n')


def main(args):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if args.infer:
        infer(args)
    else:
        train(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/training_label.txt')
    parser.add_argument('--test_path', default='data/testing_data.txt')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--weight')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

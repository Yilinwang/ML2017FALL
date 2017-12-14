from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.losses import mean_squared_error
from keras.models import load_model
import data_processor, models
import numpy as np
import pickle

from scipy.sparse.linalg import svds



def unpack(X):
    return [np.array(x) for x in zip(*X)]


def train(args):
    X, Y = getattr(data_processor, args.prepro)(args.train_path)
    model = models.get_model(args.model)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1126)

    filepath = 'model/' + args.prefix + '_' + args.model + '_{epoch:03d}_{val_mean_squared_error:.2f}'
    checkpoint = ModelCheckpoint(filepath)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks_list = [checkpoint, earlyStopping]

    history = model.fit(unpack(X_train), Y_train, validation_data = (unpack(X_test), Y_test), callbacks = callbacks_list, batch_size = 32, epochs = 100)


def vali(args):
    X, Y = getattr(data_processor, args.prepro)(args.train_path)
    model = load_model(args.weight)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1126)
    
    Y_pred = model.predict(unpack(X_test))
    print(Y_pred)
    print(mean_squared_error(Y_test, Y_pred))


def infer(args):
    X = getattr(data_processor, 'read_test_data')(args.test_path)
    model = load_model(args.weight)

    with open(args.output, 'w') as fp:
        fp.write('TestDataID,Rating\n')
        for idx, p in enumerate(model.predict(unpack(X))):
            fp.write(str(idx+1) + ',' + str(p) + '\n')


def ensemble(args):
    X_test = getattr(data_processor, 'read_test_data')(args.test_path)

    models = []
    pred = models[0].predict(X_test)

    with open(args.output, 'w') as fp:
        fp.write('id,label\n')
        for idx, p in enumerate(pred):
            print(idx, p)
            fp.write(str(idx) + ',' + str(np.argmax(p)) + '\n')


def main(args):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if args.infer:
        infer(args)
    elif args.ensemble:
        ensemble(args)
    elif args.vali:
        vali(args)
    else:
        train(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/train.csv')
    parser.add_argument('--test_path', default='data/test.csv')
    parser.add_argument('--prepro', default='read_train_data')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--vali', action='store_true')
    parser.add_argument('--weight')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

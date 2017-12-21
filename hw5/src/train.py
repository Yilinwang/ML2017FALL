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
    model = models.get_model(args.model, args.dim)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1126)

    filepath = 'model/' + args.prefix + '_' + args.model + '_{epoch:03d}_{val_mean_squared_error:.2f}'
    checkpoint = ModelCheckpoint(filepath)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks_list = [checkpoint, earlyStopping]

    history = model.fit(unpack(X_train), Y_train, validation_data = (unpack(X_test), Y_test), callbacks = callbacks_list, batch_size = 32, epochs = 100)


def loss(y, y_pred):
    return (sum([(a - b) ** 2 for a, b in zip(y, y_pred)]) / len(y)) ** (1/2)


def vali(args):
    X, Y = getattr(data_processor, args.prepro)(args.train_path)
    model = load_model(args.weight)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1126)
    
    Y_pred = model.predict(unpack(X_test))
    mean, std = pickle.load(open('model/norm_weight.pkl', 'rb'))
    print(loss(Y_test, (Y_pred * std) + mean))


def infer(args):
    X = getattr(data_processor, 'read_test_data')(args.test_path)
    model = load_model(args.weight)

    with open(args.output, 'w') as fp:
        fp.write('TestDataID,Rating\n')
        for idx, p in enumerate(model.predict(unpack(X))):
            rating = max(min(p[0], 5), 0)
            fp.write(str(idx+1) + ',' + str(rating) + '\n')


def ensemble(args):
    X_test = getattr(data_processor, 'read_test_data')(args.test_path)

    models = []
    models.append(load_model('model/8_ta_011_0.76'))
    models.append(load_model('model/4_ta_011_0.77'))
    models.append(load_model('model/8_ta_020_0.77'))
    models.append(load_model('model/8_ta_015_0.77'))
    models.append(load_model('model/4_ta_022_0.77'))
    models.append(load_model('model/4_ta_028_0.77'))

    pred = models[0].predict(unpack(X_test))
    pred += models[1].predict(unpack(X_test))
    pred += models[2].predict(unpack(X_test))
    pred += models[3].predict(unpack(X_test))
    pred += models[4].predict(unpack(X_test))
    pred += models[5].predict(unpack(X_test))

    pred = pred / 6

    with open(args.output, 'w') as fp:
        fp.write('TestDataID,Rating\n')
        for idx, p in enumerate(pred):
            rating = max(min(p[0], 5), 0)
            fp.write(str(idx+1) + ',' + str(rating) + '\n')


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
    parser.add_argument('--dim', type = int)
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

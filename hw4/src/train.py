from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1126)

    history = model.fit(x = X_train, y = Y_train, validation_data = (X_test, Y_test), epochs = 100, batch_size = 32, callbacks = callbacks_list)


def infer(args):
    X = getattr(data_processor, 'read_test_data')(args.test_path)
    model = load_model(args.weight)

    with open(args.output, 'w') as fp:
        fp.write('id,label\n')
        for idx, p in enumerate(model.predict(X)):
            fp.write(str(idx) + ',' + str(np.argmax(p)) + '\n')


def ensemble(args):
    X_test = getattr(data_processor, 'read_test_data')(args.test_path)

    models = []
    models.append(load_model('model/lff_003_0.82'))
    models.append(load_model('model/lff_004_0.82'))
    models.append(load_model('model/lff_005_0.82'))

    pred = models[0].predict(X_test)
    pred += models[1].predict(X_test)
    pred += models[2].predict(X_test)

    with open(args.output, 'w') as fp:
        fp.write('id,label\n')
        for idx, p in enumerate(pred):
            fp.write(str(idx) + ',' + str(np.argmax(p)) + '\n')


def main(args):
    if args.infer:
        infer(args)
    elif args.ensemble:
        ensemble(args)
    else:
        train(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/training_label.txt')
    parser.add_argument('--test_path', default='data/testing_data.txt')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--weight')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

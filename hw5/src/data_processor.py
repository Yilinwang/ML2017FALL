from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pickle



def norm(path):
    X = list()
    Y = list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            dataid, userid, movieid, rating = [int(x) for x in line.strip().split(',')]
            X.append((userid, movieid))
            Y.append(rating)
    return X, np.array(Y) / 5


def read_train_data(path):
    X = list()
    Y = list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            dataid, userid, movieid, rating = [int(x) for x in line.strip().split(',')]
            X.append((userid, movieid))
            Y.append(rating)
    return X, np.array(Y)


def read_test_data(path):
    X = list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            dataid, userid, movieid = [int(x) for x in line.strip().split(',')]
            X.append((userid, movieid))
    return X


if __name__ == '__main__':
    read_train_data('data/train.csv')

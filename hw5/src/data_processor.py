from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pickle



def norm(path):
    user = dict()
    for idx, line in enumerate(open('data/users.csv')):
        if idx != 0:
            tok = line.strip().split('::')
            user[int(tok[0])] = tok[1:]
    X = list()
    Y = list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            dataid, userid, movieid, rating = [int(x) for x in line.strip().split(',')]
            X.append((userid, movieid, user[userid]))
            Y.append(rating)
    Y = np.array(Y)
    pickle.dump((Y.mean(), Y.std()), open('model/norm_weight.pkl', 'wb'))
    return X, (Y - Y.mean()) / Y.std()


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
    norm('data/train.csv')

from keras.utils import to_categorical
import numpy as np
import word2vec
import pickle



def read_train_data(path):
    X = list()
    Y = list()
    '''
    wvmodel = word2vec.Word2Vec()
    pickle.dump(wvmodel, open('model/word2vec', 'wb'))
    '''
    wvmodel = pickle.load(open('model/word2vec', 'rb'))
    for idx, line in enumerate(open(path)):
        label, sentence = line.split(' +++$+++ ')
        Y.append(label)
        X.append(sentence.strip().split())
    max_len = len(max(X, key=len)) + 2
    X = np.array([wvmodel.getwv(s, max_len) for s in X])
    Y = to_categorical(np.array(Y))
    return X, Y


def read_test_data(path):
    X = list()
    wvmodel = pickle.load(open('model/word2vec', 'rb'))
    for idx, line in enumerate(open(path)):
        if idx != 0:
            sentence = line.split(',', 1)[-1]
            X.append(wvmodel.getwv(sentence, 41))
    X = np.array(X)
    return X


if __name__ == '__main__':
    read_train_data('data/training_label.txt')

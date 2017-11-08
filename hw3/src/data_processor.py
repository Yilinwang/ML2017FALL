import numpy as np



def read_train_data(path):
    X, Y = list(), list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            label, feature = line.split(',')
            X.append(feature.split(' '))
            y = np.zeros((7))
            y[int(label)] = 1
            Y.append(y)
    return np.array(X).reshape((len(X), 48, 48, 1)), np.array(Y)


def read_test_data(path):
    X = list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            feature = line.split(',')[1]
            X.append(feature.split(' '))
    return np.array(X).reshape((len(X), 48, 48, 1))
            

def process_data(train_path, test_path):
    data = dict()
    print('------ READING DATA ------')
    data['X'], data['Y'] = read_train_data(train_path)
    data['X_test'] = read_test_data(test_path)
    print('------ READING DATA FINISHED ------')
    return data

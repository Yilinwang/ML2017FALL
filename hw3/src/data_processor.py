import numpy as np



def read_train_data(path):
    X, Y = list(), list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            label, feature = line.split(',')
            X.append(np.array([float(x) for x in feature.split(' ')]))
            y = np.zeros((7))
            y[int(label)] = 1
            Y.append(y)
    X = np.array(X)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X.reshape((len(X), 48, 48, 1)), np.array(Y)


def read_test_data(path):
    X = list()
    for idx, line in enumerate(open(path)):
        if idx != 0:
            feature = line.split(',')[1]
            X.append(np.array([float(x) for x in feature.split(' ')]))
    X = np.array(X)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X.reshape((len(X), 48, 48, 1))
            

def process_data(train_path, test_path):
    data = dict()
    print('------ READING DATA ------')
    data['X'], data['Y'] = read_train_data(train_path)
    data['X_test'] = read_test_data(test_path)
    print('------ READ DATA FINISHED ------')
    return data

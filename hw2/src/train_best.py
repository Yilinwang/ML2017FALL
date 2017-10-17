from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Activation
from scipy.special import expit
from numpy.linalg import inv
import pandas as pd
import numpy as np
import keras



def process_data(args):
    df_train = pd.read_csv('data/X_train')
    df_trainY = pd.read_csv('data/Y_train')
    df_test = pd.read_csv('data/X_test')

    for x in df_train:
        if df_train[x].max() > 1 or df_train[x].min() < 0:
            df_test[x] = (df_test[x] - df_train[x].mean()) / df_train[x].std()
            df_train[x] = (df_train[x] - df_train[x].mean()) / df_train[x].std()
            #df_test = pd.concat([df_test, df_test[x] ** 2], axis = 1)
            #df_train = pd.concat([df_train, df_train[x] ** 2], axis = 1)

    #df_test = (df_test - df_train.mean()) / df_train.std()
    #df_train = (df_train - df_train.mean()) / df_train.std()

    return {'df_train': df_train, 'df_test': df_test, 'df_trainY': df_trainY}


def predict(w, x):
    return [0 if p < 0.5 else 1 for p in [expit(np.dot(w, i)) for i in x]]


def evaluation(y, y_pred):
    t = 0
    for j, k in zip(y, y_pred):
        if j == k:
            t += 1
    return t / len(y)
    


def df2feature(df):
    x = df.values
    x = np.hstack([np.ones((len(x), 1)), x]) # add bias
    return x


def nn(x, y):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(107,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x, y, epochs=10, batch_size=32)
    return model


def main(args):
    data = process_data(args)
    
    x = df2feature(data['df_train'])
    y = data['df_trainY']['label'].values
    y = keras.utils.to_categorical(y, 2)

    kf = KFold(n_splits=3)
    acc_t = 0
    for train_i, test_i in kf.split(x):
        x_train, x_test = x[train_i], x[test_i]
        y_train, y_test = y[train_i], y[test_i]
        model = nn(x_train, y_train)
        r = model.evaluate(x_test, y_test, verbose=0)
        acc_t += r[1]
        print(r)
    print(acc_t / 3)

    model = nn(x, y)
    model.predict(df2feature(data['df_test'])).dump('result/testnn')

'''
    w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
    print(evaluation(y, predict(w, x)))
    with open('result/first.csv', 'w') as fp:
        fp.write('id,label\n')
        for i, p in enumerate(predict(w, df2feature(data['df_test']))):
            fp.write(f'{i+1},{p}\n')
'''


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--trainX_path', default='./data/X_train')
    parser.add_argument('-y', '--trainY_path', default='./data/Y_train')
    parser.add_argument('-t', '--testX_path', default='./data/X_test')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

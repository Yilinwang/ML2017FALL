from scipy.special import expit
from numpy.linalg import inv
import pandas as pd
import numpy as np



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


def main(args):
    data = process_data(args)
    
    x = df2feature(data['df_train'])
    y = data['df_trainY']['label'].values

    w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
    print(evaluation(y, predict(w, x)))
    with open('result/first.csv', 'w') as fp:
        fp.write('id,label\n')
        for i, p in enumerate(predict(w, df2feature(data['df_test']))):
            fp.write(f'{i+1},{p}\n')


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

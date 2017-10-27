from scipy.special import expit
import pandas as pd
import numpy as np



def process_data(args):
    df_train = pd.read_csv('data/X_train')
    df_trainY = pd.read_csv('data/Y_train')
    df_test = pd.read_csv('data/X_test')

    df_test = pd.concat([df_test, df_test ** 2], axis = 1)
    df_train = pd.concat([df_train, df_train ** 2], axis = 1)
    df_test = pd.concat([df_test, df_test ** 3], axis = 1)
    df_train = pd.concat([df_train, df_train ** 3], axis = 1)

    df = pd.concat([df_train, df_test])

    df_test = (df_test - df.mean()) / df.std()
    df_train = (df_train - df.mean()) / df.std()

    return {'df_train': df_train, 'df_test': df_test, 'df_trainY': df_trainY}
'''
    for x in df_train:
        if df_train[x].max() > 1 or df_train[x].min() < 0:
            df_test[x] = (df_test[x] - df_train[x].mean()) / df_train[x].std()
            df_train[x] = (df_train[x] - df_train[x].mean()) / df_train[x].std()
            #df_test = pd.concat([df_test, df_test[x] ** 2], axis = 1)
            #df_train = pd.concat([df_train, df_train[x] ** 2], axis = 1)
'''


def sigmoid(x):
    ans = 1 / (1.0 + np.exp(-x))
    return np.clip(ans, 1e-8, 1-(1e-8))


def predict(w, x):
    return [0 if p < 0.5 else 1 for p in [sigmoid(np.dot(w, i)) for i in x]]


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


def gdfit(x, y, w, lr, b_size = 32):
    w_sum = np.zeros(len(x[0]))
    for i in range(len(x)):
        w_sum = w_sum + (lr) * (-2) * (y[i] - sigmoid(np.dot(w, x[i]))) * x[i]
        if i % 32 == 0:
            w -= w_sum/len(x)
            w_sum = np.zeros(len(x[0]))
    return w


def main(args):
    data = process_data(args)
    
    x = df2feature(data['df_train'])
    y = data['df_trainY']['label'].values
    w = np.ones(len(x[0])) / len(x[0])

    fp = open(f'result/log/{args.prefix}.csv', 'w')
    fp.write('epoch,training_precision\n')
    for epoch in range(args.epoch):
        w = gdfit(x, y, w, args.lr)
        precision = evaluation(y, predict(w, x))

        print(f'{epoch},{precision}')
        fp.write(f'{epoch},{precision}\n')
        with open(f'result/prediction/{args.prefix}_{epoch}.csv', 'w') as prefp:
            prefp.write('id,label\n')
            for i, p in enumerate(predict(w, df2feature(data['df_test']))):
                prefp.write(f'{i+1},{p}\n')
        w.dump(f'result/model/{args.prefix}_{epoch}.csv')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--trainX_path', default='./data/X_train')
    parser.add_argument('-y', '--trainY_path', default='./data/Y_train')
    parser.add_argument('-t', '--testX_path', default='./data/X_test')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-e', '--epoch', type=int)
    parser.add_argument('-l', '--lr', type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

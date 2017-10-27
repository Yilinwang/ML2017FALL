from scipy.special import expit
import pandas as pd
import numpy as np



def process_data(args):
    df_train = pd.read_csv(args.trainX_path)
    df_trainY = pd.read_csv(args.trainY_path)
    df_test = pd.read_csv(args.testX_path)

    df_test = pd.concat([df_test, df_test ** 2], axis = 1)
    df_train = pd.concat([df_train, df_train ** 2], axis = 1)
    df_test = pd.concat([df_test, df_test ** 3], axis = 1)
    df_train = pd.concat([df_train, df_train ** 3], axis = 1)

    df = pd.concat([df_train, df_test])

    df_test = (df_test - df.mean()) / df.std()
    df_train = (df_train - df.mean()) / df.std()

    return {'df_train': df_train, 'df_test': df_test, 'df_trainY': df_trainY}


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


def train(args):
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


def infer(args):
    data = process_data(args)
    w = np.load('model/logis')
    with open(args.output_path, 'w') as fp:
        fp.write('id,label\n')
        for i, p in enumerate(predict(w, df2feature(data['df_test']))):
            fp.write(f'{i+1},{p}\n')


def main(args):
    if args.infer:
        infer(args)
    else:
        train(args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--trainX_path')
    parser.add_argument('-y', '--trainY_path')
    parser.add_argument('-t', '--testX_path')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-e', '--epoch', type=int)
    parser.add_argument('-l', '--lr', type=float)
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

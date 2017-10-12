from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.special import expit
import pandas as pd
import numpy as np
import pickle



def process_data(train_path, test_path):
    df = (pd.read_csv(train_path, encoding='big5', parse_dates=['日期'],
                      index_col=['日期', '測站', '測項'])
          .apply(lambda s:pd.to_numeric(s, errors='conerce')).fillna(0).groupby('日期')
          .apply(lambda df: df.reset_index(['日期', '測站'], drop=True).T))

    train_df = df[(df.index.get_level_values('日期').day < 17) | (df.index.get_level_values('日期').day > 20)]
    vali_df = df[(df.index.get_level_values('日期').day >= 17) & (df.index.get_level_values('日期').day <= 20)]

    test_df = (pd.read_csv(test_path, header=None, index_col=[0, 1])
               .apply(lambda s:pd.to_numeric(s, errors='conerce')).fillna(0).groupby(0)
               .apply(lambda df: df.reset_index([0], drop=True).T))

    train_df_scaled = (train_df - df.mean()) / df.std()
    vali_df_scaled = (vali_df - df.mean()) / df.std()
    test_df_scaled = (test_df - df.mean()) / df.std()

    return train_df, vali_df, test_df, train_df_scaled, vali_df_scaled, test_df_scaled


def df2feature(df, df_scaled):
    x = []
    y = []
    for i in range(0, df.shape[0]-10):
        # overlap
        tmpdf = df[i:i+10]
        tmpdf_scaled = df_scaled[i:i+10]
        date = tmpdf.index.get_level_values('日期')
        if not (max(date) - min(date)).days > 1:
            x.append(tmpdf_scaled.values[:-1].flatten())
            y.append(tmpdf['PM2.5'].values[-1])
    x = np.array(x)
    x = np.hstack([np.ones((len(x), 1)), x])
    return x, np.array(y)


def testdf2feature(df):
    return df.groupby(level=[0]).apply(lambda df: df.stack().reset_index(0, drop=True))


def ada(x):
    return ((x ** 2).sum()) ** 0.5


def gdfit(x, y, w1, b1, w2, b2, lr):
    w1_sum = np.zeros(w1.shape)
    w2_sum = np.zeros(w2.shape)
    b1_sum = np.zeros(b1.shape)
    b2_sum = 0
    for i in np.random.choice(len(x), 128, replace=False):
        d1 = (-2) * (y[i] - h(w1, b1, w2, b2, x[i]))
        a = np.dot(w1, x[i]) + b1
        ds = dactivation(a)
        for r in range(len(w1_sum)):
            tmp = d1 * w2[r] * ds[r]
            w1_sum[r] += tmp * x[r]
            b1_sum[r] += tmp
        w2_sum += d1 * activation(a)
        b2_sum += d1
    gdfit.w1_sum += w1_sum
    gdfit.w2_sum += w2_sum
    gdfit.b1_sum += b1_sum
    gdfit.b2_sum += b2_sum
    w1 -= (lr / ada(gdfit.w1_sum)) * w1_sum
    w2 -= (lr / ada(gdfit.w2_sum)) * w2_sum
    b1 -= (lr / ada(gdfit.b1_sum)) * b1_sum
    b2 -= (lr / ada(gdfit.b2_sum)) * b2_sum
    return w1, b1, w2, b2


def dactivation(x):
    #return np.array([0 if n < 0 else 1 for n in x])
    return expit(x) * (1 - expit(x))


def activation(x):
    #return np.array([0 if n < 0 else n for n in x])
    #return x / (1 + np.absolute(x))
    return expit(x)


def h(w1, b1, w2, b2, x):
    return np.dot(w2, activation(np.dot(w1, x)) + b1) + b2


def predict(w1, b1, w2, b2, vali_x):
    return [h(w1, b1, w2, b2, x) for x in vali_x]


def evaluation(y_pred, y):
    return mean_squared_error(y, y_pred)**0.5


def main(args):
    np.random.seed(args.rseed)

    train_df, vali_df, test_df, train_df_scaled, vali_df_scaled, test_df_scaled = process_data(args.train_path, args.test_path)
    train_x, train_y = df2feature(train_df, train_df_scaled)
    vali_x, vali_y = df2feature(vali_df, vali_df_scaled)
    x = np.append(train_x, vali_x, axis=0)
    y = np.append(train_y, vali_y)

    feature_df = testdf2feature(test_df_scaled)
    feature_df.index.names = ['id']
    test_x = np.hstack([np.ones((len(feature_df.values), 1)), feature_df.values])

    fp = open(f'result_nn/csv_log/{args.prefix}_{args.lr}', 'w')
    fp.write('iter,train,vali\n')

    k = args.nlayer
    n = len(x[0])

    w1 = np.random.rand(k, n)
    w2 = np.random.rand(k)
    b1 = np.random.rand(k)
    b2 = np.random.rand(1)[0]

    gdfit.w1_sum = np.zeros(w1.shape)
    gdfit.w2_sum = np.zeros(w2.shape)
    gdfit.b1_sum = np.zeros(b1.shape)
    gdfit.b2_sum = 0

    for t in range(args.epoch):
        tmp_w = gdfit(train_x, train_y, w1, b1, w2, b2, args.lr)
        vali_rmse = evaluation(predict(w1, b1, w2, b2, vali_x), vali_y)

        w1, b1, w2, b2 = gdfit(x, y, w1, b1, w2, b2, args.lr)
        with open(f'result_nn/model/{args.prefix}_{args.lr}_{t}', 'wb') as modelfp:
            pickle.dump((w1, b1, w2, b2), modelfp)

        feature_df['value'] = predict(w1, b1, w2, b2, test_x)
        feature_df['value'].to_csv(f'result_nn/prediction/{args.prefix}_{args.lr}_{t}', header=True)
        train_rmse = evaluation(predict(w1, b1, w2, b2, x), y)
        print(f'{t},{train_rmse:.8f},{vali_rmse}')
        fp.write(f'{t},{train_rmse},{vali_rmse}\n')


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', default='./data/train.csv')
    parser.add_argument('-s', '--test_path', default='./data/test.csv')
    parser.add_argument('-e', '--epoch', default=1000, type=int)
    parser.add_argument('-l', '--lr', default=0.0001, type=float)
    parser.add_argument('-p', '--prefix', default='')
    parser.add_argument('-n', '--nlayer', default=4, type=int)
    parser.add_argument('-r', '--rseed', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_arg())

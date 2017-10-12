from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math



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
            #x.append(tmpdf_scaled[['PM2.5', 'PM10', 'NO2', 'NOx', 'SO2', 'AMB_TEMP', 'O3', 'RH']].values[:-1].flatten())
            x.append(tmpdf_scaled.values[:-1].flatten())
            y.append(tmpdf['PM2.5'].values[-1])
    x = np.array(x)
    x = np.hstack([x, x ** 2])
    x = np.hstack([np.ones((len(x), 1)), x])
    return x, np.array(y)


def testdf2feature(df):
    #return df.groupby(level=[0]).apply(lambda df: df[['PM2.5', 'PM10', 'NO2', 'NOx', 'SO2', 'AMB_TEMP', 'O3', 'RH']].stack().reset_index(0, drop=True))
    return df.groupby(level=[0]).apply(lambda df: df.stack().reset_index(0, drop=True))


def gdfit(x, y, w, lr):
    w_sum = np.zeros(len(x[0]))
    for i in range(len(x)):
        w_sum = w_sum + (lr) * (-2) * (y[i] - np.dot(w, x[i])) * x[i]
    w -= w_sum/len(x)
    return w


def predict(w, vali_x):
    return [np.dot(w, x) for x in vali_x]


def evaluation(y_pred, y):
    return mean_squared_error(y, y_pred)**0.5


def main(args):
    train_df, vali_df, test_df, train_df_scaled, vali_df_scaled, test_df_scaled = process_data(args.train_path, args.test_path)
    train_x, train_y = df2feature(train_df, train_df_scaled)
    vali_x, vali_y = df2feature(vali_df, vali_df_scaled)
    x = np.append(train_x, vali_x, axis=0)
    y = np.append(train_y, vali_y)

    feature_df = testdf2feature(test_df_scaled)
    feature_df.index.names = ['id']
    test_x = feature_df.values
    test_x = np.hstack([test_x, test_x ** 2])
    test_x = np.hstack([np.ones((len(feature_df.values), 1)), test_x])

    fp = open(f'result/csv_log/{args.prefix}_{args.lr}', 'w')
    fp.write('iter,train,vali\n')
    w = np.ones(len(train_x[0])) / len(train_x[0])

    for t in range(args.epoch):
        tmp_w = gdfit(train_x, train_y, w, args.lr)
        vali_rmse = evaluation(predict(tmp_w, vali_x), vali_y)

        w = gdfit(x, y, w, args.lr)
        w.dump(f'result/model/{args.prefix}_{args.lr}_{t}') 

        feature_df['value'] = predict(w, test_x)
        feature_df['value'].to_csv(f'result/prediction/{args.prefix}_{args.lr}_{t}', header=True)
        train_rmse = evaluation(predict(w, x), y)
        print(f'{t},{train_rmse:.8f},{vali_rmse}')
        fp.write(f'{t},{train_rmse},{vali_rmse}\n')


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', default='./data/train.csv')
    parser.add_argument('-s', '--test_path', default='./data/test.csv')
    parser.add_argument('-e', '--epoch', default=10000, type=int)
    parser.add_argument('-l', '--lr', default=0.001, type=float)
    parser.add_argument('-p', '--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_arg())

import pandas as pd
import numpy as np
import pickle
import math



def testdf2feature(df):
    return df.groupby(level=[0]).apply(lambda df: df.stack().reset_index(0, drop=True))


def predict(w, vali_x):
    return [np.dot(w, x) for x in vali_x]


def main(args):
    with open('model/scale.pkl', 'wb') as fp:
        std, mean = pickle.load(fp)

    w = np.load('model/best')

    test_df = (pd.read_csv(args.test_path, header=None, index_col=[0, 1])
               .apply(lambda s:pd.to_numeric(s, errors='conerce')).fillna(0).groupby(0)
               .apply(lambda df: df.reset_index([0], drop=True).T))
    test_df_scaled = (test_df - mean) / std

    feature_df = testdf2feature(test_df_scaled)
    feature_df.index.names = ['id']
    test_x = feature_df.values
    test_x = np.hstack([test_x, test_x ** 2])
    test_x = np.hstack([np.ones((len(feature_df.values), 1)), test_x])

    feature_df['value'] = predict(w, test_x)
    feature_df['value'].to_csv(args.output_path, header=True)


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path')
    parser.add_argument('-i', '--test_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_arg())

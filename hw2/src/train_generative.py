from scipy.special import expit
import pandas as pd
import numpy as np
import pickle



def process_data(args):
    df_train = pd.read_csv('data/X_train')
    df_trainY = pd.read_csv('data/Y_train')
    df_test = pd.read_csv('data/X_test')

    for x in df_train:
        if df_train[x].max() > 1 or df_train[x].min() < 0:
            df_test[x] = (df_test[x] - df_train[x].mean()) / df_train[x].std()
            df_train[x] = (df_train[x] - df_train[x].mean()) / df_train[x].std()

    return {'df_train': df_train, 'df_test': df_test, 'df_trainY': df_trainY}


def sigmoid(x):
    ans = 1 / (1.0 + np.exp(-x))
    return np.clip(ans, 1e-8, 1-(1e-8))


def evaluation(y, y_pred):
    t = 0
    for j, k in zip(y, y_pred):
        if j == k:
            t += 1
    return t / len(y)


def df2feature(df):
    x = df.values
    #x = np.hstack([np.ones((len(x), 1)), x]) # add bias
    return x


def gdfit(x, y, w, lr, b_size = 32):
    w_sum = np.zeros(len(x[0]))
    for i in range(len(x)):
        w_sum = w_sum + (lr) * (-2) * (y[i] - sigmoid(np.dot(w, x[i]))) * x[i]
        if i % 32 == 0:
            w -= w_sum/len(x)
            w_sum = np.zeros(len(x[0]))
    return w


def maxl(x, y):
    train_data_size = x.shape[0]
    dim = len(x[0])

    cnt1 = 0
    cnt2 = 0
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    for i in range(train_data_size):
        if y[i] == 1:
            mu1 += x[i]
            cnt1 += 1
        else:
            mu2 += x[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(train_data_size):
        if y[i] == 1:
            sigma1 += np.dot(np.transpose([x[i]-mu1]), [x[i]-mu1])
        else:
            sigma2 += np.dot(np.transpose([x[i]-mu2]), [x[i]-mu2])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    return {'shared_sigma': shared_sigma, 'mu1': mu1, 'mu2': mu2, 'N1': cnt1, 'N2': cnt2}


def predict(model, x):
    sigma_inverse = np.linalg.inv(model['shared_sigma'])
    w = np.dot((model['mu1']-model['mu2']), sigma_inverse)
    x = x.T
    b = (-0.5) * np.dot(np.dot([model['mu1']], sigma_inverse), model['mu1']) + (0.5) * np.dot(np.dot([model['mu2']], sigma_inverse), model['mu2']) + np.log(float(model['N1'])/model['N2'])
    return np.around(sigmoid(np.dot(w, x) + b))


def train(args):
    data = process_data(args)
    
    x = df2feature(data['df_train'])
    y = data['df_trainY']['label'].values

    model = maxl(x, y)
    precision = evaluation(y, predict(model, x))

    print(f'{precision}')
    with open(f'result/prediction/{args.prefix}.csv', 'w') as prefp:
        prefp.write('id,label\n')
        for i, p in enumerate(predict(model, df2feature(data['df_test']))):
            prefp.write(f'{i+1},{int(p)}\n')
    with open(f'result/model/{args.prefix}.csv', 'wb') as fp:
        pickle.dump(model, fp)


def infer(args):
    data = process_data(args)
    model = pickle.load(open('model/gene', 'rb'))
    with open(args.output_path, 'w') as prefp:
        prefp.write('id,label\n')
        for i, p in enumerate(predict(model, df2feature(data['df_test']))):
            prefp.write(f'{i+1},{int(p)}\n')


def main(args):
    if args.infer:
        infer(args)
    else:
        train(args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument('-x', '--trainX_path', default='./data/X_train')
    parser.add_argument('-y', '--trainY_path', default='./data/Y_train')
    parser.add_argument('-t', '--testX_path', default='./data/X_test')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-e', '--epoch', type=int)
    parser.add_argument('-l', '--lr', type=float)
    '''
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

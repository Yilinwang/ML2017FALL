from sklearn.model_selection import KFold
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from scipy.special import expit
from numpy.linalg import inv
import pandas as pd
import numpy as np
import keras



def process_data(args):
    df_train = pd.read_csv(args.trainX_path)
    df_trainY = pd.read_csv(args.trainY_path)
    df_test = pd.read_csv(args.testX_path)

    for x in df_train:
        if df_train[x].max() > 1 or df_train[x].min() < 0:
            df_test[x] = (df_test[x] - df_train[x].mean()) / df_train[x].std()
            df_train[x] = (df_train[x] - df_train[x].mean()) / df_train[x].std()

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
    return model


def save2file(log, path):
    with open(path, 'w') as fp:
        fp.write(log)


def train(args):
    data = process_data(args)
    
    x = df2feature(data['df_train'])
    y = data['df_trainY']['label'].values
    y = keras.utils.to_categorical(y, 2)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model = nn(x, y)
    history = model.fit(x, y, validation_split=0.2, epochs=10, batch_size=32)
    log = f"acc: {history.history['acc'][-1]:.4f}, val_acc:{history.history['val_acc'][-1]:.4f}"
    prediction = 'id,label\n'
    for idx, c in enumerate([0 if i[0] > i[1] else 1 for i in model.predict(df2feature(data['df_test']))]):
        prediction += f'{idx+1},{c}\n'

    model.save(f'result/model/{args.prefix}')
    save2file(prediction, f'result/prediction/{args.prefix}.csv')
    save2file(log, f'result/log/{args.prefix}')


def infer(args):
    data = process_data(args)
    model = load_model('model/best')
    with open(args.output_path, 'w') as prefp:
        prefp.write('id,label\n')
        for idx, c in enumerate([0 if i[0] > i[1] else 1 for i in model.predict(df2feature(data['df_test']))]):
            prefp.write(f'{idx+1},{c}\n')


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

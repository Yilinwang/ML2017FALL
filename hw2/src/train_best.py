import sys
sys.path.append('../tools/')
import gmail

from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def save2file(log, path):
    with open(path, 'w') as fp:
        fp.write(log)


def main(args):
    data = process_data(args)
    
    x = df2feature(data['df_train'])
    y = data['df_trainY']['label'].values
    y = keras.utils.to_categorical(y, 2)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model = nn(x, y)
    history = model.fit(x, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])
    log = f"acc: {history.history['acc'][-1]:.4f}, val_acc:{history.history['val_acc'][-1]:.4f}"
    prediction = 'id,label\n'
    for idx, c in enumerate([0 if i[0] > i[1] else 1 for i in model.predict(df2feature(data['df_test']))]):
        prediction += f'{idx+1},{c}\n'

    model.save(f'result/model/{args.prefix}')
    save2file(prediction, f'result/prediction/{args.prefix}.csv')
    save2file(log, f'result/log/{args.prefix}')

    email_config = {
        'from': 'yentingg.lee@gmail.com',
        'to': 'linda.yilin@gmail.com',
        'password': '54UT52!)',
        'mail_server': 'smtp.gmail.com:587',
    }
    gmail.email_message(email_config, f'result of {args.prefix}', content=log, email_data=[{'name': 'prediction.csv', 'data': prediction}])


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--trainX_path', default='./data/X_train')
    parser.add_argument('-y', '--trainY_path', default='./data/Y_train')
    parser.add_argument('-t', '--testX_path', default='./data/X_test')
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

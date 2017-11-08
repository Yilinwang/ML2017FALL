from keras.callbacks import ModelCheckpoint, EarlyStopping



def parse_config():
    config = dict()
    with open('config') as fp:
        config['model'] = fp.read().strip()
    return config


def train(args):
    model_config = parse_config['model']

    import data_processor, cnn_model
    data = getattr(data_processor, 'process_data')(args.train_path, args.test_path)
    model = getattr(cnn_model, model_config)()

    filepath = f'result/model/{model_config}_'
    filepath += '{epoch:03d}'
    checkpoint = ModelCheckpoint(filepath)

    earlyStopping = EarlyStopping(patience=3)

    callbacks_list = [checkpoint, earlyStopping]
    history = model.fit(data['X'], data['Y'], validation_split=0.2, epochs=1000, batch_size=32, callbacks=callbacks_list)


def main(args):
    train(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--test_path', default='./data/test.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

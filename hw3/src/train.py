from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np



def train(args):
    import data_processor, cnn_model
    data = getattr(data_processor, 'process_data')(args.train_path, args.test_path)
    model = cnn_model.get_model(args.model)

    filepath = 'model/' + args.prefix + args.model + '_{epoch:03d}_{val_acc:.2f}'
    checkpoint = ModelCheckpoint(filepath)

    earlyStopping = EarlyStopping(monitor='loss', patience=3)

    callbacks_list = [checkpoint, earlyStopping]
    history = model.fit(data['X'], data['Y'], validation_split=0.2, epochs=100, batch_size=128, callbacks=callbacks_list)


def infer(args):
    import data_processor, cnn_model
    data = getattr(data_processor, 'process_data')(args.train_path, args.test_path)
    model = cnn_model.get_model(args.model, args.weight)

    with open(args.output, 'w') as fp:
        fp.write('id,label\n')
        for idx, p in enumerate(model.predict(data['X_test'])):
            fp.write(str(idx) + ',' + str(np.argmax(p)) + '\n')


def main(args):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))

    if args.infer:
        infer(args)
    else:
        train(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--test_path', default='./data/test.csv')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--weight')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np



def train(args):
    import data_processor, cnn_model
    data = getattr(data_processor, 'process_data')(args.train_path, args.test_path)
    model = cnn_model.get_model(args.model)

    filepath = 'model/' + args.prefix + args.model + '_{epoch:03d}_{val_acc:.2f}'
    checkpoint = ModelCheckpoint(filepath)

    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, turewise_std_normalization=True

    earlyStopping = EarlyStopping(monitor='loss', patience=3)

    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, zoom_range=0.2)
    vali_datagen = ImageDataGenerator()

    callbacks_list = [checkpoint, earlyStopping]
    #history = model.fit(data['X'], data['Y'], validation_split=0.1, epochs=100, batch_size=32, callbacks=callbacks_list)
    
    vali_size = data['X'].shape[0] // 10
    train_X, train_Y = data['X'][:-vali_size], data['Y'][:-vali_size]
    vali_X, vali_Y = data['X'][-vali_size:], data['Y'][-vali_size:]
    history = model.fit_generator(datagen.flow(train_X, train_Y, batch_size=32), validation_data=vali_datagen.flow(vali_X, vali_Y), steps_per_epoch=len(data['X']) / 32, validation_steps= vali_X.shape[0] / 32, epochs=100, callbacks=callbacks_list)


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
    config.gpu_options.allow_growth = True
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

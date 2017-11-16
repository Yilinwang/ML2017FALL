from keras.utils.vis_utils import plot_model
from keras.models import load_model



def main():
    model = load_model('model/ccaccaccaccaff_da_074_0.67')
    model.summary()
    plot_model(model, to_file='model.png')


if __name__ == '__main__':
    main()

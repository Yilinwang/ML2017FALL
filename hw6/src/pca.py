from skimage import io
import numpy as np
import glob
import os



def saveimg(m, name):
    m -= np.min(m)
    m /= np.max(m)
    m = (m * 255).astype(np.uint8)
    m = m.reshape(600,600,3)
    io.imsave(name, m)


def main(args):
    X = list()
    for image in glob.glob(args.image_path + '/*.jpg'):
        img = io.imread(image)
        X.append(img.flatten())
    X = np.array(X)
    X_mean = np.mean(X, axis=0)
    U, s, V = np.linalg.svd((X - X_mean).transpose(), full_matrices=False)
    U = U.transpose()

    y = io.imread(os.path.join(args.image_path, args.image_name))
    y = y.flatten()
    y = y - X_mean
    W = np.dot(U[:4], y)
    img = X_mean
    for w, u in zip(W, U[:4]):
        img += w * u
    saveimg(img, 'reconstruction.jpg')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='./Aberdeen')
    parser.add_argument('--image_name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

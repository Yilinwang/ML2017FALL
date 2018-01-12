from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pickle



def main(args):
    if args.infer:
        infer(args)
    else:
        images = np.load(args.image_path)

        pca = PCA(n_components=300, whiten=True)
        X = pca.fit_transform(images)

        kmeans = KMeans(n_clusters=2, random_state=1005).fit(X)

        pickle.dump(kmeans.labels_, open('model.pkl', 'wb'))

        with open(args.output, 'w') as wfp:
            wfp.write('ID,Ans\n')
            for idx, line in enumerate(open(args.test_path)):
                if idx != 0:
                    tok = line.strip().split(',')
                    if kmeans.labels_[int(tok[1])] == kmeans.labels_[int(tok[2])]:
                        wfp.write(tok[0] + ',1\n')
                    else:
                        wfp.write(tok[0] + ',0\n')
    return


def infer(args):
    label = pickle.load(open('model.pkl', 'rb'))
    with open(args.output, 'w') as wfp:
        wfp.write('ID,Ans\n')
        for idx, line in enumerate(open(args.test_path)):
            if idx != 0:
                tok = line.strip().split(',')
                if label[int(tok[1])] == label[int(tok[2])]:
                    wfp.write(tok[0] + ',1\n')
                else:
                    wfp.write(tok[0] + ',0\n')
    return


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='data/image.npy')
    parser.add_argument('--test_path', default='data/test_case.csv')
    parser.add_argument('--output')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--n_c', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())

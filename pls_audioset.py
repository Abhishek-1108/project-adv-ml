from glob import glob
import math
import os
import pickle

from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import ipdb


def get_video_features(basedir):
    X = []
    # ipdb.set_trace()
    ids = []
    for f in glob(basedir + '/*.npz'):
        ytid = os.path.splitext(os.path.split(f)[1])[0]
        ids.append(ytid)
        fullpath = os.path.join(basedir, f)
        with open(fullpath) as infile:
            x = np.load(infile)

        x = x.flatten()
        X.append(x)

    X = np.asarray(X)
    return X, ids


def get_audio_features(basedir, ids):
    X = []
    required_count = 10
    for id in ids:
        fullpath = os.path.join(basedir, id) + ".pkl"
        with open(fullpath) as infile:
            x = pickle.load(infile)

        x_length = len(x)
        if x_length < required_count:
            diff = required_count - x_length
            half_diff = (diff * 1.0) / 2

            start_rep = int(math.floor(half_diff))
            end_rep = int(math.ceil(half_diff))
            # pad the array with edges,
            # if the length of the frameset is less than required
            # only pad so as to repeat the frame (axis 0),
            # and not the dimensions themselves (axis 1)
            x = np.lib.pad(x, ((start_rep, end_rep), (0, 0)), 'edge')

        x = x.flatten()
        X.append(x)

    X = np.asarray(X)
    return X


def pls_decomposition(videos, audios, n_components=256):
    plsca = PLSCanonical(n_components=n_components)
    plsca.fit(audios, videos)

    videos_c, audios_c = plsca.transform(videos, audios)
    return videos_c, audios_c


def save_outputs(output_dir, ids, videos, audios, videos_pca, videos_c, audios_c):

    with open(output_dir + 'ids.pkl', 'wb') as outfile:
        pickle.dump(ids, outfile)

    with open(output_dir + 'videos.pkl', 'wb') as outfile:
        pickle.dump(videos, outfile)

    with open(output_dir + 'audios.pkl', 'wb') as outfile:
        pickle.dump(audios, outfile)

    with open(output_dir + 'videos_pca.pkl', 'wb') as outfile:
        pickle.dump(videos_pca, outfile)

    with open(output_dir + 'videos_c.pkl', 'wb') as outfile:
        pickle.dump(videos_c, outfile)

    with open(output_dir + 'audios_c.pkl', 'wb') as outfile:
        pickle.dump(audios_c, outfile)

    fused = []
    for i in range(videos_c.shape[0]):
        fused.append(
            np.hstack((videos_c[i], audios_c[i]))
        )

    fused = np.asarray(fused)
    print('Fused shape {}'.format(fused.shape))

    with open(output_dir + 'fused.pkl', 'wb') as outfile:
        pickle.dump(fused, outfile)


def reduce_dimensions(X, n_components):
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    Xdash = svd.transform(X)
    return Xdash, svd


def main():
    video_basedir = '/proj/audioset_video_features/'
    audio_basedir = '/proj/pickles_audioset/bal_train/'
    output_dir = '/proj/pls_audioset_experiments/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videos, ids = get_video_features(video_basedir)
    print("Videos shape {}".format(videos.shape))

    audios = get_audio_features(audio_basedir, ids)
    print("Audios shape {}".format(audios.shape))

    # original video data is sparse, so we can hopefully do much better with PCA
    videos_transformed, v_reducer = reduce_dimensions(videos, 512)
    print('Videos Transformed shape {}'.format(videos_transformed.shape))
    audios_transformed, a_reducer = reduce_dimensions(audios, 512)
    print('Audios Transformed shape {}'.format(audios_transformed.shape))

    videos_c, audios_c = pls_decomposition(videos_transformed, audios_transformed)
    print('PLS shapes {} {}'.format(videos_c.shape, audios_c.shape))

    save_outputs(output_dir, ids, videos, audios, videos_transformed, videos_c, audios_c)


if __name__ == '__main__':
    main()

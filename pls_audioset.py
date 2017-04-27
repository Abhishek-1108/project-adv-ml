from glob import glob
import os
import pickle

from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
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
    for id in ids:
        fullpath = os.path.join(basedir, id[:2])
        with open(fullpath) as infile:
            x = pickle.load(infile)

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
    for i in videos_c.shape[0]:
        fused.append(
            np.hstack((videos_c[i], audios_c[i]))
        )

    fused = np.asarray(fused)
    print('Fused shape {}'.format(fused.shape))

    with open(output_dir + 'fused.pkl', 'wb') as outfile:
        pickle.dump(fused, outfile)


def main():
    video_basedir = '/Users/mridul/Downloads/features/'  # '/proj/audioset_video_features/'
    audio_basedir = '/proj/audioset_v1_pickles/'
    output_dir = '/proj/pls_audioset_experiments/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videos, ids = get_video_features(video_basedir)
    print("Videos shape {}".format(videos.shape))

    audios = get_audio_features(audio_basedir, ids)
    print("Audios shape {}".format(audios.shape))

    # original video data is sparse, so we can hopefully do much better with PCA
    pca = PCA(n_components=1280)
    pca.fit(videos)
    videos_transformed = pca.transform(videos)
    print("Videos Transformed shape {}".format(videos_transformed.shape))

    videos_c, audios_c = pls_decomposition(videos, audios)
    print('PLS shapes {} {}'.format(videos_c.shape, audios_c.shape))

    save_outputs(output_dir, ids, videos, audios, videos_transformed, videos_c, audios_c)


if __name__ == '__main__':
    main()

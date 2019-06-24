
import os
import numpy as np
from contextlib import ExitStack
from glob import glob
import pickle
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.preprocessing import minmax, meanvar

from hparams import hparams as hp


class FeatureFileSource(FileDataSource):
    def __init__(self, data_root, dim):
        self.data_root = data_root
        self.dim = dim

    def collect_files(self):
        files = sorted(glob(os.path.join(self.data_root, "*.npy")))
        files = files[:len(files)-5]  # last 5 is real testset
        return files

    def collect_features(self, path):
        return np.load(path).reshape(-1, self.dim)


DATA_ROOT = os.path.join(os.getcwd(), 'datasets/slt_arctic_full_data')

X = {"duration": [], "acoustic": []}
Y = {"duration": [], "acoustic": []}
utt_lengths = {"duration": [], "acoustic": []}
for ty in ["duration", "acoustic"]:
    x_dim = hp.duration_linguistic_dim if ty == "duration" else hp.acoustic_linguistic_dim
    y_dim = hp.duration_dim if ty == "duration" else hp.acoustic_dim
    X[ty] = FileSourceDataset(FeatureFileSource(os.path.join(DATA_ROOT, "X_{}".format(ty)),
                                                dim=x_dim))
    Y[ty] = FileSourceDataset(FeatureFileSource(os.path.join(DATA_ROOT, "Y_{}".format(ty)),
                                                dim=y_dim))
    # this triggers file loads, but can be neglectable in terms of performance.
    utt_lengths[ty] = [len(x) for x in X[ty]]

X_min = {}
X_max = {}
Y_mean = {}
Y_var = {}
Y_scale = {}

for typ in ["acoustic", "duration"]:
    X_min[typ], X_max[typ] = minmax(X[typ], utt_lengths[typ])
    Y_mean[typ], Y_var[typ] = meanvar(Y[typ], utt_lengths[typ])
    Y_scale[typ] = np.sqrt(Y_var[typ])

fname_list = ['X_min.pkl', 'X_max.pkl', 'Y_mean.pkl', 'Y_var.pkl', 'Y_scale.pkl']

with ExitStack() as stack:
    f = [stack.enter_context(open(os.path.join(DATA_ROOT, fname), 'wb')) for fname in fname_list]
    pickle.dump(X_min, f[0])
    pickle.dump(X_max, f[1])
    pickle.dump(Y_mean, f[2])
    pickle.dump(Y_var, f[3])
    pickle.dump(Y_scale, f[4])
    print("Finished!")




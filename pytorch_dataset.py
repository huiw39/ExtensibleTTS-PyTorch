
import os
import pickle
import numpy as np
from contextlib import ExitStack
import torch
from torch.utils.data import Dataset

from nnmnkwii.preprocessing import minmax_scale, scale

from hparams import hparams as hp


class FeatureDataset(Dataset):

    def __init__(self, data_path, metadata, X_min, X_max, Y_mean, Y_scale, train):
        # with open(os.path.join(data_path, 'dataset_ids.pkl'), 'rb') as fid:
        #     self.metadata = pickle.load(fid)
        self.metadata = metadata
        self.X_min = X_min
        self.X_max = X_max
        self.Y_mean = Y_mean
        self.Y_scale = Y_scale
        if train == 'duration':
            self.x_dim = hp.duration_linguistic_dim
            self.y_dim = hp.duration_dim
        else:
            self.x_dim = hp.acoustic_linguistic_dim
            self.y_dim = hp.acoustic_dim
        self.train = train
        self.X_path = os.path.join(data_path, 'X_{}'.format(train))
        self.Y_path = os.path.join(data_path, 'Y_{}'.format(train))

    def __getitem__(self, index):
        file = self.metadata[index]
        x = np.load(os.path.join(self.X_path, '{}.npy'.format(file))).reshape(-1, self.x_dim)
        y = np.load(os.path.join(self.Y_path, '{}.npy'.format(file))).reshape(-1, self.y_dim)
        norm_x = minmax_scale(x, self.X_min[self.train], self.X_max[self.train], feature_range=(0.01, 0.99))
        norm_y = scale(y, self.Y_mean[self.train], self.Y_scale[self.train])
        return norm_x, norm_y

    def __len__(self):
        return len(self.metadata)


def dnn_collate(batch):

    input_lengths = [len(x[0]) for x in batch]
    max_len = np.max(input_lengths)
    x = [_pad_2d(x[0], max_len) for x in batch]
    y = [_pad_2d(x[1], max_len) for x in batch]
    x_batch = np.stack(x).astype(np.float32)
    y_batch = np.stack(y).astype(np.float32)

    return torch.FloatTensor(x_batch), torch.FloatTensor(y_batch)


def rnn_collate(batch):

    input_lengths = [len(x[0]) for x in batch]
    max_len = np.max(input_lengths)
    x = [_pad_2d(x[0], max_len) for x in batch]
    y = [_pad_2d(x[1], max_len) for x in batch]
    x_batch = np.stack(x).astype(np.float32)
    y_batch = np.stack(y).astype(np.float32)

    return torch.FloatTensor(x_batch), torch.LongTensor(input_lengths), torch.FloatTensor(y_batch)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)], mode="constant", constant_values=0)
    return x








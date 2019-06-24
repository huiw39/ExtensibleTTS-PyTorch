
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import numpy as np
from tqdm import tqdm
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe


def build_from_path(in_dir, out_dir, label_type=None, num_workers=1, tqdm=lambda x: x):

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_feature on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'file_id_list_full.scp'), encoding='utf-8') as f:
        for line in f:
            label_path = os.path.join(in_dir, 'label_{}'.format(label_type), '%s.lab' % line.strip())
            futures.append(executor.submit(partial(_process_feature, out_dir, index, label_path)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_feature(out_dir, index, label_path):

    labels = hts.load(label_path)
    features = fe.duration_features(labels)
    n_frames = len(features)
    indices = labels.silence_phone_indices()
    features = np.delete(features, indices, axis=0)
    voiced_frames = features.shape[0]

    # Write the duration to disk:
    duration_filename = 'arctic_%05d.npy' % index
    np.save(os.path.join(out_dir, duration_filename), features.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (duration_filename, n_frames, voiced_frames)

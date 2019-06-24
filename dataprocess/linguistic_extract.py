
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import numpy as np
from tqdm import tqdm
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe


def build_from_path(in_dir, out_dir,
                    add_frame_features=False, subphone_features=None,
                    question_path=None, label_type=None, num_workers=1,
                    tqdm=lambda x: x):

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_feature on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'file_id_list_full.scp'), encoding='utf-8') as f:
        for line in f:
            label_path = os.path.join(in_dir, 'label_{}'.format(label_type), '%s.lab' % line.strip())
            futures.append(executor.submit(partial(_process_feature, out_dir, index, label_path,
                                                   add_frame_features=add_frame_features,
                                                   subphone_features=subphone_features,
                                                   question_path=question_path)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_feature(out_dir, index, label_path,
                     add_frame_features=False, subphone_features=None, question_path=None):

    labels = hts.load(label_path)
    binary_dict, continuous_dict = hts.load_question_set(question_path)
    features = fe.linguistic_features(labels,
                                      binary_dict, continuous_dict,
                                      add_frame_features=add_frame_features,
                                      subphone_features=subphone_features)
    n_frames = len(features)
    if add_frame_features:
        indices = labels.silence_frame_indices().astype(np.int)
    else:
        indices = labels.silence_phone_indices()
    features = np.delete(features, indices, axis=0)
    voiced_frames = features.shape[0]

    # Write the linguistic to disk:
    linguistic_filename = 'arctic_%05d.npy' % index
    np.save(os.path.join(out_dir, linguistic_filename), features.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (linguistic_filename, n_frames, voiced_frames)

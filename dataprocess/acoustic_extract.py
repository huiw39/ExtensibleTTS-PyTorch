
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import numpy as np
from tqdm import tqdm
import pickle
from scipy.io import wavfile
import pyworld
import pysptk
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnmnkwii.io import hts


def build_from_path(in_dir, out_dir, label_type=None, num_workers=1, tqdm=lambda x: x):

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_feature on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'file_id_list_full.scp'), encoding='utf-8') as f:
        for line in f:
            wav_path = os.path.join(in_dir, 'wav', '%s.wav' % line.strip())
            label_path = os.path.join(in_dir, 'label_{}'.format(label_type), '%s.lab' % line.strip())
            futures.append(executor.submit(partial(_process_feature, out_dir, index, wav_path, label_path)))
            index += 1
    return [future.result() for future in tqdm(futures)]

  
order = 59
frame_period = 5
windows = [
  (0, 0, np.array([1.0])),
  (1, 1, np.array([-0.5, 0.0, 0.5])),
  (1, 1, np.array([1.0, -2.0, 1.0])),
]

dataset_ids = []


def _process_feature(out_dir, index, wav_path, label_path):

    # get list of wav files
    wav_files = os.listdir(os.path.dirname(wav_path))
    # check wav_file
    assert len(wav_files) != 0 and wav_files[0][-4:] == '.wav', "no wav files found!"

    fs, x = wavfile.read(wav_path)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    n_frames = len(f0)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    bap = pyworld.code_aperiodicity(aperiodicity, fs)
    mgc = pysptk.sp2mc(spectrogram, order=order, alpha=pysptk.util.mcepalpha(fs))
    f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    vuv = (lf0 != 0).astype(np.float32)
    lf0 = interp1d(lf0, kind="slinear")

    mgc = apply_delta_windows(mgc, windows)
    lf0 = apply_delta_windows(lf0, windows)
    bap = apply_delta_windows(bap, windows)

    features = np.hstack((mgc, lf0, vuv, bap))

    # get list of lab files
    lab_files = os.listdir(os.path.dirname(label_path))
    # check wav_file
    assert len(lab_files) != 0 and lab_files[0][-4:] == '.lab', "no lab files found!"

    # Cut silence frames by HTS alignment
    labels = hts.load(label_path)
    features = features[:labels.num_frames()]
    indices = labels.silence_frame_indices()
    features = np.delete(features, indices, axis=0)
    voiced_frames = features.shape[0]

    # Write the acoustic to disk:
    acoustic_filename = 'arctic_%05d.npy' % index
    np.save(os.path.join(out_dir, acoustic_filename), features.astype(np.float32), allow_pickle=False)

    dataset_ids.append(acoustic_filename[:-4])
    with open(os.path.join(os.path.dirname(out_dir), 'dataset_ids.pkl'), 'wb') as pklFile:
        pickle.dump(dataset_ids, pklFile)

    # Return a tuple describing this training example:
    return (acoustic_filename, n_frames, voiced_frames)

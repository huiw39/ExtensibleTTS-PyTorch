
import argparse
import os
import numpy as np
from contextlib import ExitStack
import pickle
from scipy.io import wavfile
import torch
import torch.nn as nn

from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.postfilters import merlin_post_filter
import pyworld
import pysptk

from hparams import hparams as hp
from model import DurationModel, AcousticModel, RNNet

fftlen = pyworld.get_cheaptrick_fft_size(hp.fs)
alpha = pysptk.util.mcepalpha(hp.fs)
hop_length = int(0.001 * hp.frame_period * hp.fs)
windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]


def gen_parameters(y_predicted, Y_var):
    # Number of time frames
    T = y_predicted.shape[0]

    # Split acoustic features
    mgc = y_predicted[:, :hp.lf0_start_idx]
    lf0 = y_predicted[:, hp.lf0_start_idx:hp.vuv_start_idx]
    vuv = y_predicted[:, hp.vuv_start_idx]
    bap = y_predicted[:, hp.bap_start_idx:]

    # Perform MLPG
    ty = "acoustic"
    mgc_variances = np.tile(Y_var[ty][:hp.lf0_start_idx], (T, 1))
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(Y_var[ty][hp.lf0_start_idx:hp.vuv_start_idx], (T, 1))
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(Y_var[ty][hp.bap_start_idx:], (T, 1))
    bap = paramgen.mlpg(bap, bap_variances, windows)

    return mgc, lf0, vuv, bap


def gen_waveform(y_predicted, Y_var, do_postfilter=False):

    y_predicted = trim_zeros_frames(y_predicted)

    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted, Y_var)
    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), hp.fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            hp.fs, hp.frame_period).astype(np.int16)
    return generated_waveform


def gen_duration(device, label_path, binary_dict, continuous_dict,
                 X_min, X_max, Y_mean, Y_scale, duration_model):

    # Linguistic features for duration
    hts_labels = hts.load(label_path)
    duration_linguistic_features = fe.linguistic_features(hts_labels,
                                                          binary_dict, continuous_dict,
                                                          add_frame_features=False,
                                                          subphone_features=None).astype(np.float32)
    # Apply normalization
    ty = "duration"
    duration_linguistic_features = minmax_scale(
        duration_linguistic_features, X_min[ty], X_max[ty], feature_range=(0.01, 0.99))

    # # Apply model
    # # duration_model = duration_model.cpu()
    duration_model.eval()
    x = torch.FloatTensor(duration_linguistic_features)
    duration_predicted = duration_model(x.unsqueeze(0)).data.numpy()
    print("duration_predicted shape: {}".format(duration_predicted.shape))

    # Apply denormalization
    duration_predicted = duration_predicted * Y_scale[ty] + Y_mean[ty]
    duration_predicted = np.round(duration_predicted)

    # Set minimum state duration to 1
    duration_predicted[duration_predicted <= 0] = 1
    hts_labels.set_durations(duration_predicted)

    return hts_labels


def lab2wav(args, device, label_path, binary_dict, continuous_dict,
            X_min, X_max, Y_mean, Y_var, Y_scale,
            duration_model, acoustic_model, post_filter=False):
    # Predict durations
    duration_modified_hts_labels = gen_duration(device, label_path, binary_dict, continuous_dict,
                                                X_min, X_max, Y_mean, Y_scale, duration_model)

    # Linguistic features
    linguistic_features = fe.linguistic_features(duration_modified_hts_labels,
                                                 binary_dict, continuous_dict,
                                                 add_frame_features=True,
                                                 subphone_features="full" if args.label == 'state_align' else "coarse_coding")

    # Trim silences
    indices = duration_modified_hts_labels.silence_frame_indices()
    linguistic_features = np.delete(linguistic_features, indices, axis=0)

    # Apply normalization
    ty = "acoustic"
    linguistic_features = minmax_scale(linguistic_features,
                                       X_min[ty], X_max[ty], feature_range=(0.01, 0.99))

    # Predict acoustic features
    # acoustic_model = acoustic_model.cpu()
    acoustic_model.eval()
    x = torch.FloatTensor(linguistic_features)
    acoustic_predicted = acoustic_model(x.unsqueeze(0)).data.numpy()
    print("acoustic_predicted shape: {}".format(acoustic_predicted.shape))

    # Apply denormalization
    acoustic_predicted = acoustic_predicted * Y_scale[ty] + Y_mean[ty]

    return gen_waveform(acoustic_predicted.squeeze(0), Y_var, post_filter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser(os.getcwd()))
    parser.add_argument('--data', default='datasets/slt_arctic_full_data')
    parser.add_argument('--label', required=True, choices=['state_align', 'phone_align'])
    parser.add_argument('--question', default='questions-radio_dnn_416.hed')
    parser.add_argument('--duration_checkpoint', required=True, help='Path to duration model checkpoint')
    parser.add_argument('--acoustic_checkpoint', required=True, help='Path to acoustic model checkpoint')

    args = parser.parse_args()
    data_root = os.path.join(args.base_dir, args.data)
    save_dir = os.path.join(data_root, 'generate')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    fname_list = ['X_min.pkl', 'X_max.pkl', 'Y_mean.pkl', 'Y_var.pkl', 'Y_scale.pkl']
    with ExitStack() as stack:
        f = [stack.enter_context(open(os.path.join(data_root, fname), 'rb')) for fname in fname_list]
        X_min = pickle.load(f[0])
        X_max = pickle.load(f[1])
        Y_mean = pickle.load(f[2])
        Y_var = pickle.load(f[3])
        Y_scale = pickle.load(f[4])

    binary_dict, continuous_dict = hts.load_question_set(os.path.join(data_root, args.question))

    # Build model
    duration_model = DurationModel(
        hp.duration_linguistic_dim, hp.hidden_size, hp.duration_dim, hp.num_layers)
    acoustic_model = AcousticModel(
        hp.acoustic_linguistic_dim, hp.hidden_size, hp.acoustic_dim, hp.num_layers)
    # duration_model = RNNet(
    #     hp.duration_linguistic_dim, hp.hidden_size, hp.duration_dim, hp.num_layers, bidirectional=True)
    # acoustic_model = RNNet(
    #     hp.acoustic_linguistic_dim, hp.hidden_size, hp.acoustic_dim, hp.num_layers, bidirectional=True)

    # Load checkpoint
    duration_log_dir = os.path.join(data_root, 'logs-duration')
    duration_checkpoint_path = os.path.join(duration_log_dir, args.duration_checkpoint)
    duration_checkpoint = torch.load(duration_checkpoint_path)
    duration_model.load_state_dict(duration_checkpoint["state_dict"])
    print("loading duration model from checkpoint:{}".format(duration_checkpoint_path))

    acoustic_log_dir = os.path.join(data_root, 'logs-acoustic')
    acoustic_checkpoint_path = os.path.join(acoustic_log_dir, args.acoustic_checkpoint)
    acoustic_checkpoint = torch.load(acoustic_checkpoint_path)
    acoustic_model.load_state_dict(acoustic_checkpoint["state_dict"])
    print("loading acoustic model from checkpoint:{}".format(acoustic_checkpoint_path))

    # Label to waveform
    label_dir = os.path.join(data_root, 'label_{}'.format(args.label))
    test_labels = os.listdir(label_dir)[::-1][:5][::-1]
    for label in test_labels:
        label_path = os.path.join(label_dir, label)
        wav_file = os.path.basename(label)[:-4] + '.wav'
        waveform = lab2wav(args, device, label_path, binary_dict, continuous_dict,
                           X_min, X_max, Y_mean, Y_var, Y_scale, duration_model, acoustic_model, post_filter=True)
        wavfile.write(os.path.join(save_dir, wav_file), rate=hp.fs, data=waveform)

    print("Finished! Check out {} for synthesis audio samples.".format(save_dir))


if __name__ == '__main__':
    main()


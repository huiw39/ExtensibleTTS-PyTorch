import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from dataprocess import linguistic_extract, duration_extract, acoustic_extract
  

def Duration_LinguisticExtract(args, 
                               add_frame_features=False, subphone_features=None):
    in_dir = os.path.join(args.base_dir, args.data)
    out_dir = os.path.join(in_dir, args.dur_in)
    os.makedirs(out_dir, exist_ok=True)
    question_path = os.path.join(in_dir, args.question)
    metadata = linguistic_extract.build_from_path(in_dir, out_dir,
                                                  add_frame_features, subphone_features,
                                                  question_path, args.label, args.num_workers,
                                                  tqdm=tqdm)
    write_metadata(metadata, out_dir)
  
  
def DurationExtract(args):
    in_dir = os.path.join(args.base_dir, args.data)
    out_dir = os.path.join(in_dir, args.dur_out)
    os.makedirs(out_dir, exist_ok=True)
    metadata = duration_extract.build_from_path(in_dir, out_dir, args.label, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)
  
  
def Acoustic_LinguisticExtract(args, 
                               add_frame_features=False, subphone_features=None):
    in_dir = os.path.join(args.base_dir, args.data)
    out_dir = os.path.join(in_dir, args.acous_in)
    os.makedirs(out_dir, exist_ok=True)
    question_path = os.path.join(in_dir, args.question)
    metadata = linguistic_extract.build_from_path(in_dir, out_dir,
                                                  add_frame_features, subphone_features,
                                                  question_path, args.label, args.num_workers,
                                                  tqdm=tqdm)
    write_metadata(metadata, out_dir)
  
  
def AcousticExtract(args):
    in_dir = os.path.join(args.base_dir, args.data)
    out_dir = os.path.join(in_dir, args.acous_out)
    os.makedirs(out_dir, exist_ok=True)
    metadata = acoustic_extract.build_from_path(in_dir, out_dir, args.label, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[1] for m in metadata])
    voiced_frames = sum([m[2] for m in metadata])
    hours = frames * 5 / (3600 * 1000)
    voiced_hours = voiced_frames * 5 / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f houts), %d voiced_frames (%.2f voiced_hours)' %
          (len(metadata), frames, hours, voiced_frames, voiced_hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser(os.getcwd()))
    parser.add_argument('--data', default='datasets/slt_arctic_full_data')
    parser.add_argument('--label', required=True, choices=['state_align', 'phone_align'])
    parser.add_argument('--question', default='questions-radio_dnn_416.hed')
    parser.add_argument('--dur_in', default='X_duration')
    parser.add_argument('--dur_out', default='Y_duration')
    parser.add_argument('--acous_in', default='X_acoustic')
    parser.add_argument('--acous_out', default='Y_acoustic')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
  
    args = parser.parse_args()
  
    Duration_LinguisticExtract(args, add_frame_features=False, subphone_features=None)
    DurationExtract(args)
    if args.label == 'state_align':
        Acoustic_LinguisticExtract(args, add_frame_features=True, subphone_features="full")
    else:
        Acoustic_LinguisticExtract(args, add_frame_features=True, subphone_features="coarse_coding")
    AcousticExtract(args)


if __name__ == "__main__":
    main()

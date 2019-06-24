
import os
import argparse
import pickle
from contextlib import ExitStack
import torch

from hparams import hparams as hp
from utils import infolog
from pytorch_dataset import FeatureDataset, dnn_collate
from model import DurationModel, AcousticModel
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay

global_step = 0
global_epoch = 0

log = infolog.log

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   


def save_checkpoint(device, model, optimizer, step, epoch, log_dir):
    checkpoint_path = os.path.join(log_dir, "model.ckpt-{}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = _load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(checkpoint_path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


def train_loop(device, model, optimizer, data_loader, log_dir):

    global global_step, global_epoch

    criterion = torch.nn.MSELoss().to(device)

    while global_epoch < hp.epochs:
        running_loss = 0
        for i, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            # calculate learning rate and update learning rate
            if hp.fixed_learning_rate:
                current_lr = hp.fixed_learning_rate
            elif hp.lr_schedule_type == 'step':
                current_lr = step_learning_rate_decay(hp.init_learning_rate, global_step,
                                                      hp.step_gamma, hp.lr_step_interval)
            else:
                current_lr = noam_learning_rate_decay(hp.init_learning_rate, global_step,
                                                      hp.noam_warm_up_steps)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()                                             # gradient clear
            loss.backward()                                                   # BP for gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)  # clip gradient norm
            optimizer.step()                                                  # update weight parameter

            running_loss += loss.item()
            avg_loss = running_loss / (i+1)

            # saving checkpoint
            if global_step != 0 and global_step % hp.checkpoint_interval == 0:
                # pruner.prune(global_step)
                save_checkpoint(device, model, optimizer, global_step, global_epoch, log_dir)
            global_step += 1

        print("epoch:{:4d}  [loss={:.5f}, avg_loss={:.5f}, current_lr={}]".format(global_epoch,
                                                                                  running_loss, avg_loss,
                                                                                  current_lr))
        global_epoch += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser(os.getcwd()))
    parser.add_argument('--data', default='datasets/slt_arctic_full_data')
    parser.add_argument('--train_model', required=True, choices=['duration', 'acoustic'])
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')

    # Parameter analysis
    args = parser.parse_args()
    data_root = os.path.join(args.base_dir, args.data)
    run_name = args.name or args.train_model
    log_dir = os.path.join(data_root, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train.log'), run_name)

    fname_list = ['dataset_ids.pkl', 'X_min.pkl', 'X_max.pkl', 'Y_mean.pkl', 'Y_scale.pkl']
    with ExitStack() as stack:
        f = [stack.enter_context(open(os.path.join(data_root, fname), 'rb')) for fname in fname_list]
        metadata = pickle.load(f[0])
        X_min = pickle.load(f[1])
        X_max = pickle.load(f[2])
        Y_mean = pickle.load(f[3])
        Y_scale = pickle.load(f[4])

    train_set = FeatureDataset(data_root, metadata, X_min, X_max, Y_mean, Y_scale, train=run_name)
    data_loader = torch.utils.data.DataLoader(train_set,
                                              collate_fn=dnn_collate,
                                              batch_size=hp.batch_size,
                                              shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Build model, create optimizer
    if args.train_model == 'duration':
        model = DurationModel(
            hp.duration_linguistic_dim, hp.hidden_size, hp.duration_dim, hp.num_layers).to(device)
    else:
        model = AcousticModel(
            hp.acoustic_linguistic_dim, hp.hidden_size, hp.acoustic_dim, hp.num_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp.init_learning_rate,
                                 betas=(hp.adam_beta1, hp.adam_beta2),
                                 eps=hp.adam_eps,
                                 weight_decay=hp.weight_decay,
                                 amsgrad=hp.amsgrad)

    # Reload parameters from a checkpoint
    if args.restore_step:
        checkpoint_path = os.path.join(log_dir, 'model.ckpt-{}.pth'.format(args.restore_step))
        model = load_checkpoint(checkpoint_path, model, optimizer, False)
        print("Resuming from checkpoint:{}".format(checkpoint_path))

    # Train loop
    try:
        train_loop(device, model, optimizer, data_loader, log_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        print("Saving checkpoint....")
        save_checkpoint(device, model, optimizer, global_step, global_epoch, log_dir)


if __name__ == '__main__':
    main()

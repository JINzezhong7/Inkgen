import argparse
import PyInk
import Texts
import sys
import os
import numpy as np
import torch
import random
sys.path.append(r"../src")
from generate import generate_teacherforcing
from utilz import offset_points2strokes
import h5py
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import jiwer
import string

def test(args, train_loader, save_path, prefix, char_to_code_file, model_path, priming_x=None, priming_text="", x_r=None):
    os.makedirs(os.path.join(save_path, 'pngs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'isfs'), exist_ok=True)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    for i, (data, masks, onehots) in enumerate(train_loader):
        data , masks, onehots = data.to(device), masks.to(device), onehots.to(device)
        points = generate_teacherforcing(data,
                                onehots,
                                save_name=os.path.join(save_path,"pngs",f"{prefix}_{i}.png"),
                                device=torch.device('cpu'),
                                cell_size=512,
                                num_clusters=20,
                                K=10,
                                z_size=0,
                                char_to_code_file=char_to_code_file,
                                state_dict_file=model_path,
                                bias=1.0, bias2=1.0,
                                priming_x=priming_x,
                                priming_text=priming_text,
                                x_r=x_r,
                                model_type = 'transformer'
            )





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../small_training_data',
                        help='directory to load training data')
    parser.add_argument('--model_dir', type=str, default='../save',
                        help='directory to save model to')
    parser.add_argument('--cell_size', type=int, default=512,
                        help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='rms',
                        help='optimizer to use (rms or adam)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='whether or not to use LR scheduler')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='number of warmup steps')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='lr decay rate for adam optimizer per epoch')
    parser.add_argument('--num_clusters', type=int, default=20,
                        help='number of gaussian mixture clusters for stroke prediction')
    parser.add_argument('--K', type=int, default=10,
                        help='number of attention clusters on text input')
    parser.add_argument('--z_size', type=int, default=256,
                        help='style distribution size')
    parser.add_argument('--clip_value', type=float, default=100,
                        help='value to which to clip non-LSTM gradients')
    parser.add_argument('--lstm_clip_value', type=float, default=10,
                        help='value to which to clip LSTM gradients')
    parser.add_argument('--resume_from_ckpt', type=str, default=None,
                        help='checkpoint path from which to resume training')
    parser.add_argument('--style_equalization', action='store_true',
                        help='whether or not to train with style equalization')
    parser.add_argument('--model_type', type=str, default='transformer', 
                        help='model to use (alexrnn or transformer)')
    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='gloo')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        enable_dist = True
    else:
        local_rank = 0
        enable_dist = False

    load = lambda filepath: torch.from_numpy(np.load(filepath)).type(torch.FloatTensor)
    # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
    # 76 related to SE Structure
    def impose_min_length(data):
        min_idxs = torch.where(data[1].sum(-1) >= 76)[0]
        return [data[0][min_idxs], data[1][min_idxs], data[2][min_idxs]]
    
    # prepare training data
    train_data = [load(f'{args.data_dir}/train_strokes_700.npy'), load(f'{args.data_dir}/train_masks_700.npy'), load(f'{args.data_dir}/train_onehot_700.npy')]
    train_data = impose_min_length(train_data)
    train_data = [(train_data[0][i], train_data[1][i], train_data[2][i]) for i in range(len(train_data[0]))] 
    if enable_dist:
        train_sampler = torch.utils.data.DistributedSampler(train_data, num_replicas= dist.get_world_size() , rank=dist.get_rank())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler = train_sampler, pin_memory = True)#, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)#, drop_last=True)


    test(args,
        train_loader,
        "../repo_results/teacher_forcing",
        "no_priming",
        '../small_training_data/char_to_code.pt',
        '../models/epoch_1000.pt'
            )

if __name__ == '__main__':
    main()



import argparse
import os
import time
from utils import *
from dataloader import BSDS_VOCLoader
from torch.utils.data import DataLoader
from hed import HED
import torch

def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='results/savedir', 
            help='path to save result and checkpoint')
    parser.add_argument('--datadir', type=str, default='../data', 
            help='dir to the dataset')
    parser.add_argument('--epochs', type=int, default=20, 
            help='number of total epochs to run')
    parser.add_argument('--iter-size', type=int, default=24, 
            help='number of samples in each iteration')
    parser.add_argument('--lr', type=float, default=0.005, 
            help='initial learning rate for all weights')
    parser.add_argument('--opt', type=str, default='adam', 
            help='optimizer')
    parser.add_argument('-j', '--workers', type=int, default=4, 
            help='number of data loading workers')
    parser.add_argument('--eta', type=float, default=0.3, 
            help='threshold to determine the ground truth (the eta parameter in the paper)')
    parser.add_argument('--lmbda', type=float, default=1.1, 
            help='weight on negative pixels (the beta parameter in the paper)')
    return parser.parse_args()

def main():
    args = GetArgs()
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.use_cuda = torch.cuda.is_available()
    model = HED()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)
    else:
        raise TypeError("Please use a correct optimizer in [adam, sgd]")

    ### Transfer to cuda devices
    if args.use_cuda:
        model = model.cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    ### Load Data
    train_dataset = BSDS_VOCLoader(root=args.datadir, split="train", threshold=args.eta)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.workers, shuffle=True)

    args.start_epoch = 0
    ### Train
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    for epoch in range(args.start_epoch, args.epochs):
        # train
        lr = [group['lr'] for group in optimizer.param_groups]
        avg_loss = train(train_loader, model, optimizer, epoch, args, lr)
        scheduler.step(avg_loss)
    # save model with fp16
    model.half()
    filename = 'hed.pth'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    model_filename = os.path.join(args.savedir, filename )
    torch.save(model.state_dict(), model_filename)
    return


def train(train_loader, model, optimizer, epoch, args, running_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    ## Switch to train mode
    model.train()

    end = time.time()
    iter_step = 0
    counter = 0
    batch_loss_value = 0
    print_freq = len(train_loader)//args.iter_size//10
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_loader):
        ## Measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            ## Compute output
            outputs = model(image)
            loss = 0
            for o in outputs:
                loss += cross_entropy_loss_RCF(o, label, args.lmbda)

            if not torch.all(torch.isfinite(loss)):
                print('find nan loss: ', loss)
                continue

        counter += 1
        batch_loss_value += loss.item()
        loss = loss / args.iter_size
        scaler.scale(loss).backward()
        if counter == args.iter_size:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 9)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            counter = 0
            iter_step += 1

            # record loss
            losses.update(batch_loss_value, args.iter_size)
            batch_time.update(time.time() - end)
            end = time.time()
            batch_loss_value = 0

            # display and logging
            if iter_step % print_freq == 0:
                runinfo = str(('Epoch: [{0}/{1}][{2}/{3}]\t{4}\tLoss {5}\tlr {6}\t').format(
                              epoch, args.epochs, iter_step, len(train_loader)//args.iter_size, time.asctime(),
                              losses.avg, running_lr))
                print(runinfo)
    return losses.avg

if __name__ == '__main__':
    main()


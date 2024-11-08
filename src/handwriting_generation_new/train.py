import argparse
import time
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

# import pytorch modules
import torch
import torch.optim as optim
import torch.utils.data

# import model and utilities
from model import LSTMSynthesis
from utilz import get_init_state, save_checkpoint

# find gpu
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')

eps = 1E-20 # to prevent numerical error
inf_clip = 1E+20
pi_term = -torch.Tensor([2*np.pi]).to(device).log()

# training objective
def log_likelihood(end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks, mu_z=None, log_var_z=None, mu_prior=None, log_var_prior=None, A=None, num_samples=100):
    batch_size = end.shape[0]
    
    # targets
    y_0, y_1, y_2 = torch.chunk(y, 3, dim=-1)
        
    # end of stroke prediction
    end_v = torch.clip(y_0*end + (1-y_0)*(1-end), min=eps, max=1.0)
    end_loglik = end_v.log().squeeze()

    # stop prediction
    stop_labels = torch.zeros_like(y_0).to(device)
    stop_labels[torch.arange(y_0.shape[0]), masks.sum(-1).int()-1, 0] = 1
    stop_v = torch.clip(stop_labels*stop + (1-stop_labels)*(1-stop), min=eps, max=1.0)
    stop_loglik = stop_v.log().squeeze()
    # new stroke point prediction    
    z = (y_1 - mu_1)**2/(log_sigma_1.exp()**2)\
        + ((y_2 - mu_2)**2/(log_sigma_2.exp()**2)) \
        - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 + log_sigma_2).exp())
    safe_v = torch.clip(1 - rho**2 + eps, min=eps)
    mog_lik1 =  pi_term -log_sigma_1 - log_sigma_2 - 0.5*(safe_v.log())
    mog_lik2 = z/(2*safe_v)
    mog_v = (weights.log() + (mog_lik1 - mog_lik2)).exp().sum(dim=-1)
    mog_v[torch.isnan(mog_v)] = 0.0
    mog_v = torch.clip(mog_v, min=eps, max=inf_clip)
    mog_loglik = mog_v.log()
    
    loglik = (end_loglik*masks).sum() / batch_size + (stop_loglik*masks).sum() / batch_size + (mog_loglik*masks).sum() / batch_size

    if mu_z is not None:
        kl_div = 0.5 * ((1 + log_var_z - log_var_prior - (mu_z - mu_prior).pow(2) / log_var_prior.exp() - log_var_z.exp() / log_var_prior.exp()) * masks.unsqueeze(2).repeat(1, 1, mu_z.shape[2])).sum()
        loglik += kl_div / batch_size

        trace = sum([e.T.matmul(A.T).matmul(A).matmul(A.T).matmul(A).matmul(e).squeeze() for e in torch.randn(num_samples, A.shape[-1], 1).cuda()]) / num_samples
        loglik -= trace

    return loglik


def get_style_references(x, masks, dataset):
    x_r, masks_r = torch.clone(x), torch.clone(masks)
    for i in range(x_r.shape[0]):
        if random.uniform(0, 1) > 0.5: continue
        
        datum_b, mask_b, _ = dataset[random.randint(0, len(dataset) - 1)]
        x_r[i], masks_r[i] = datum_b[:-1], mask_b
    
    return x_r, masks_r


def train(args, train_loader, validation_loader):
    # infer vocab len
    vocab_len = train_loader.dataset[0][2].shape[1]
    
    # define model and optimizer
    model = LSTMSynthesis(vocab_len, args.cell_size, args.num_clusters, args.K, args.z_size if args.style_equalization else 0).to(device)

    if args.optimizer == 'rms':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, alpha=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = None
    if args.use_scheduler:
        lr_lambda = lambda e: args.warmup_steps**0.5 * min(e**-0.5 if e != 0 else 0, e * args.warmup_steps**-1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    start = 0
    if args.resume_from_ckpt is not None:
        start = int(args.resume_from_ckpt.split('_')[-1].replace('.pt', ''))
        ckpt = torch.load(args.resume_from_ckpt)
        
        #ckpt = torch.load('save_pureeng_full_rms_continuewithlr0.0001flat_2\\epoch_120.pt')
        #for n, p in ckpt['model'].items():
        #    if 'weight_ih' in n and 'lstm1' not in n:
        #        new_p = torch.randn(p.shape[0], p.shape[1] + args.z_size)
        #        new_p[:, :p.shape[1]] = p
        #        ckpt['model'][n] = new_p

        model.load_state_dict(ckpt['model'])
        
        if args.use_scheduler:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

    writer = SummaryWriter(args.model_dir)
    
    # training
    start_time = time.time()
    for epoch in range(start, args.num_epochs):
        model.train()
        train_loss = 0
        for i, (data, masks, onehots) in enumerate(train_loader):
            data , masks, onehots = data.to(device), masks.to(device), onehots.to(device)
            # prep inputs
            batch_size = data.shape[0]
            state1, state2, state3 = get_init_state(batch_size, args.cell_size, squeeze=True), get_init_state(batch_size, args.cell_size), get_init_state(batch_size, args.cell_size)
            kappa = torch.zeros(batch_size, args.K).to(device)

            x, y = data[:, :-1], data[:, 1:]
            w = onehots[:, :1].squeeze()
            
            # feed forward
            if args.style_equalization:
                x_r, masks_r = get_style_references(x, masks, train_loader.dataset)
                outputs = model(x, onehots, w, kappa, state1, state2, state3, x_r=x_r, masks=masks, masks_r=masks_r)
                end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, _, _, _, _, _, _, mu_z, log_var_z, mu_prior, log_var_prior, A = outputs
                loss = -log_likelihood(end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks, mu_z, log_var_z, mu_prior, log_var_prior, A)#/batch_size
            else:
                outputs = model(x, onehots, w, kappa, state1, state2, state3)
                end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, _, _, _, _, _, _ = outputs
                loss = -log_likelihood(end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)#/batch_size#/torch.sum(masks)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"bad loss encountered at epoch {epoch + 1}, batch {i + 1}")
                continue
            
            train_loss += loss.item()
            print(f'\rEpoch {epoch + 1:4} {i + 1}/{len(train_loader)} batches. Loss {(train_loss / (i + 1)):7.2f}.  Curr Loss {(loss.item()):7.2f}.', end='')

            # compute grads
            optimizer.zero_grad()
            loss.backward()
            norm = model.clip_grad(args.clip_value, args.lstm_clip_value)
            print(norm)
            if  norm > args.clip_value*100:
                print("Force abnormal gradients to smaller values.")
                torch.nn.utils.clip_grad_norm_(model.parameters(), min([1.0, args.lstm_clip_value/10]))
            
            # gradient step
            optimizer.step()
            if args.use_scheduler: scheduler.step()
            
        # validation
        model.eval() 
        validation_loss = 0
        for i, (data, masks, onehots) in enumerate(validation_loader):  
            # prep inputs
            data , masks, onehots = data.to(device), masks.to(device), onehots.to(device)
            batch_size = data.shape[0]
            state1, state2, state3 = get_init_state(batch_size, args.cell_size, squeeze=True), get_init_state(batch_size, args.cell_size), get_init_state(batch_size, args.cell_size)
            kappa = torch.zeros(batch_size, args.K).to(device)
        
            x, y = data[:, :-1], data[:, 1:]
            w = onehots[:, :1].squeeze()
    
            if args.style_equalization:
                x_r, masks_r = get_style_references(x, masks, train_loader.dataset)
                outputs = model(x, onehots, w, kappa, state1, state2, state3, x_r=x_r, masks=masks, masks_r=masks_r)
                end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, _, _, _, _, _, _, mu_z, log_var_z, mu_prior, log_var_prior, A = outputs
                loss = -log_likelihood(end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks, mu_z, log_var_z, mu_prior, log_var_prior, A)#/batch_size
            else:
                outputs = model(x, onehots, w, kappa, state1, state2, state3)
                end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, _, _, _, _, _, _ = outputs
                loss = -log_likelihood(end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)#/batch_size#/torch.sum(masks)
                
            validation_loss += loss.item()

        print(f'\rEpoch {epoch + 1:4} Loss {(train_loss / len(train_loader)):7.2f}, Val loss {validation_loss / len(validation_loader):7.2f}.')
        
        writer.add_scalar("loss/train", train_loss / len(train_loader), epoch)
        writer.add_scalar("loss/val", validation_loss / len(validation_loader), epoch)
        
        # checkpoint model and training
        filename = 'epoch_{}.pt'.format(epoch + 1)
        save_checkpoint(epoch, model, validation_loss / len(validation_loader), optimizer, scheduler, args.model_dir, filename)
        
        print('wall time: {}s'.format(time.time()-start_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='directory to load training data')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--cell_size', type=int, default=512,
                        help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=100,
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
    args = parser.parse_args()
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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)#, drop_last=True)
        
    # prepare validation data
    validation_data = [load(f'{args.data_dir}/validation_strokes_700.npy'), load(f'{args.data_dir}/validation_masks_700.npy'), load(f'{args.data_dir}/validation_onehot_700.npy')]
    validation_data = impose_min_length(validation_data)
    validation_data = [(validation_data[0][i], validation_data[1][i], validation_data[2][i]) for i in range(len(validation_data[0]))] 
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size)#, shuffle=False, drop_last=True)
    
    # training
    train(args, train_loader, validation_loader)


if __name__ == '__main__':
    main()

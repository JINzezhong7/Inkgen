import argparse
import torch

from model import LSTMSynthesis
from utilz import get_init_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char_to_code_file', type=str, default='datasets\\data_inkwell_preprocessed\\char_to_code.pt',
                        help='path to char-to-code file')
    parser.add_argument('--state_dict_file', type=str, default='',
                        help='path to model checkpoint')
    parser.add_argument('--cell_size', type=int, default=512,
                        help='size of LSTM hidden state')
    parser.add_argument('--num_clusters', type=int, default=20,
                        help='number of gaussian mixture clusters for stroke prediction')
    parser.add_argument('--K', type=int, default=10,
                        help='number of attention clusters on text input')
    parser.add_argument('--z_size', type=int, default=256,
                        help='style distribution size')
    parser.add_argument('--style_equalization', action='store_true',
                        help='whether or not to train with style equalization')
    parser.add_argument('--output_name', type=str, default='hwg',
                        help='ONNX output filename')
    args = parser.parse_args()

    device = torch.device('cpu')

    
    char_to_code = torch.load(args.char_to_code_file, weights_only=True)
    model = LSTMSynthesis(len(char_to_code), args.cell_size, args.num_clusters, args.K, z_size=args.z_size if args.style_equalization else 0)
    model.load_state_dict(torch.load(args.state_dict_file)['model'])

    x, onehot, w, kappa, state1, state2, state3 = torch.randn((1, 1, 3)), torch.randn((30, len(char_to_code))), torch.randn((1, len(char_to_code))), torch.randn((1, args.K)), get_init_state(1, args.cell_size, device=device, squeeze=True), get_init_state(1, args.cell_size,device=device), get_init_state(1, args.cell_size, device=device)
    inps = (x, onehot, w, kappa, state1, state2, state3)
    
    input_names = ('x', 'onehot', 'w', 'kappa', 'state1.1_in', 'state1.2_in', 'state2.1_in', 'state2.2_in', 'state3.1_in', 'state3.2_in')
    output_names = ('end', 'stop', 'weights', 'mu_1', 'mu_2', 'log_sigma_1', 'log_sigma_2', 'rho', 'w', 'kappa', 'state1.1_out', 'state1.2_out', 'state2.1_out', 'state2.2_out', 'state3.1_out', 'state3.2_out', 'phi')
    dynamic_axes = {'onehot': {0: 'string_len'}, 'phi': {1: 'string_len'}}

    if args.style_equalization:
        x_r = torch.randn((1, 300, 3))
        inps += (x_r, None, None, 1)
        input_names += ('x_r', 'bias')
        dynamic_axes['x_r'] = {1: 'time'}
    else:
        inps += (None, None, None, 1)
        input_names += ('bias',)
        
    torch.onnx.export(model, inps, args.output_name + '.onnx', input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=14)

if __name__ == '__main__':
    main()
    
import numpy as np
import torch
from utilz import get_init_state, plot_stroke
from model import LSTMSynthesis
import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = (12,6)

# find gpu 
# device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

def generate_conditionally(text, save_name, device, cell_size=400, num_clusters=20, K=10, z_size=0, random_state=700, bias=0., bias2=0., char_to_code_file='data/char_to_code.pt', state_dict_file='trained_models/conditional_epoch_60.pt', priming_x = None, priming_text = '', x_r=None):
    
    char_to_code = torch.load(char_to_code_file, weights_only=True)
    np.random.seed(random_state)
    
    #text = text + ' '
    if priming_text != '':
        text = priming_text + ' ' + text
    
    model = LSTMSynthesis(len(char_to_code), cell_size, num_clusters, K, z_size=z_size)
    model.load_state_dict(torch.load(state_dict_file, weights_only=True)['model'])
    model = model.to(device)
        
    onehot = torch.zeros((len(text), len(char_to_code))).to(device)
    for i, c in enumerate(text):
        onehot[i][char_to_code[c]] = 1
    
    x = torch.zeros((1,1,3)).to(device)
    state1, state2, state3 = get_init_state(1, cell_size, device=device, squeeze=True), get_init_state(1, cell_size, device=device), get_init_state(1, cell_size, device=device)
    kappa = torch.zeros(1, K).to(device)
    w = onehot[:1]
    
    if priming_x is not None:
        outputs = model(priming_x, onehot, w, kappa, state1, state2, state3, x_r=x_r)
        end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, state1, state2, state3, phi = outputs
    
    points, phis = [], []
    stop, count = 0, 0
    while count <= 900:#stop <= 0.5:# and count > 20:    
        outputs = model(x, onehot, w, kappa, state1, state2, state3, x_r=x_r, bias=bias)
        end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, state1, state2, state3, phi = outputs
        
        squeeze_and_detach = lambda x: x.squeeze().cpu().detach()
        end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho = tuple(map(squeeze_and_detach, [end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho]))

        #bernoulli sample
        sample_end = int(end > 0.5)#np.random.binomial(1, end)

        #mog sample
        sample_index = np.random.choice(range(20), p=weights.numpy())
        mu = np.array([mu_1[sample_index], mu_2[sample_index]])
        sigma_1 = (log_sigma_1 - bias2)[sample_index].exp()
        sigma_2 = (log_sigma_2 - bias2)[sample_index].exp()
        c = rho[sample_index] * sigma_1 * sigma_2
        cov = np.array([[sigma_1**2, c],[c, sigma_2**2]])
        sample_point = np.random.multivariate_normal(mu, cov)
        
        out = np.insert(sample_point,0,sample_end)
        points.append(out)
        
        phi = phi.squeeze(0).data.cpu().numpy()
        phis.append(phi)

        x = torch.from_numpy(out).type(torch.FloatTensor).to(device).view(1, 1, 3)        
        count += 1
        
        #if count > 700:
        if count >=20 and np.max(phi) == phi[-1] and sample_end:
            break
        
    points = np.array(points)
    if priming_x is not None:        
        priming_x = priming_x.squeeze(0).cpu().detach().numpy()
        points_a, points_b = np.zeros((priming_x.shape[0], 4)), np.ones((points.shape[0], 4))
        points_a[:, :3], points_b[:, :3] = priming_x, points
        points = np.vstack([points_a, points_b])
    
    plot_stroke(points, save_name=save_name)
    #attention_plot(np.stack(phis).T)
    return points


def attention_plot(phis):
    phis = phis/(np.sum(phis, axis = 0, keepdims=True))
    plt.xlabel('handwriting generation')
    plt.ylabel('text scanning')
    plt.imshow(phis, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()

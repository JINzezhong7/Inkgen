import numpy as np
import torch
from utilz import get_init_state, plot_stroke
from model import LSTMSynthesis
from model_transformer import TransformerSynthesis
import matplotlib.pyplot as plt 
import os
plt.rcParams["figure.figsize"] = (12,6)

# find gpu 
# device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

def generate_conditionally(data, onehots, save_name, device, cell_size=400, num_clusters=20, K=10, z_size=0, random_state=700, bias=0., bias2=0., char_to_code_file='data/char_to_code.pt', state_dict_file='trained_models/conditional_epoch_60.pt', priming_x = None, priming_text = '', x_r=None, model_type = None):
    
    char_to_code = torch.load(char_to_code_file, weights_only=True)
    np.random.seed(random_state)
    #text = text + ' '
    if priming_text != '':
        text = priming_text + ' ' + text

    if model_type == 'alexrnn':
        model = LSTMSynthesis(len(char_to_code), cell_size, num_clusters, K, z_size=z_size)
    elif model_type == 'transformer':
        model = TransformerSynthesis(len(char_to_code), cell_size, num_clusters, K, z_size=z_size)
    # model.load_state_dict(torch.load(state_dict_file, weights_only=True)['model'])
    model.load_state_dict(torch.load(state_dict_file, weights_only=True, map_location=torch.device('cpu'))['model'])

    model = model.to(device)
    data = data.to(device)
    onehot = onehots.to(device)
    x = torch.zeros((1,1,3)).to(device)
    state1, state2, state3 = get_init_state(1, cell_size, device=device, squeeze=True), get_init_state(1, cell_size, device=device), get_init_state(1, cell_size, device=device)
    kappa = torch.zeros(1, K).to(device)
    w = onehot[:, :1, :].squeeze(1)
    
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
    
    # 分割points 并插入segments
    # segments = np.split(points, np.where(points[:, 0] == 1)[0] + 1)[:-1]
    # processed_segments = add_points_to_sparse_segments(segments, min_points=4, scale=0.1)
    # points = np.vstack(processed_segments)  # 合并 segments

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


def add_points_to_sparse_segments(segments, min_points=3, scale=0.005):
    processed_segments = []
    
    for segment in segments:
        if len(segment) < min_points:
            # 插值点的数量
            n_additional = min_points - len(segment)
            
            # 获取最后一个 `end` 状态不为 1 的点
            last_non_end_point = segment[np.where(segment[:, 0] != 1)[0][-1]]
            
            new_points = []
            for _ in range(n_additional):
                # 插入新点
                new_dx = np.random.uniform(-scale, scale)
                new_dy = np.random.uniform(-scale, scale)
                new_point = [0.0, new_dx, new_dy]  # 保持落笔状态为 0
                new_points.append(new_point)
            
            # 插入点放在 `1` 点之前
            segment = np.vstack([segment[:-1], new_points, segment[-1:]])
        
        # 确保最后一个点状态为 1（保持不变）
        processed_segments.append(segment)
    
    return processed_segments


def generate_teacherforcing(model, data, onehots, device, cell_size, K, save_path, random_state=700,bias=0.,bias2=0., x_r=None):
    np.random.seed(random_state)
    os.makedirs(os.path.join(save_path, 'pngs'), exist_ok=True)
    save_name = os.path.join(save_path,"pngs")
    model = model.to(device)
    B,T,_ = data.shape

    data = data.to(device)
    onehots = onehots.to(device)

    state1, state2, state3 = get_init_state(1, cell_size, device=device, squeeze=True), None, None
    kappa = torch.zeros(1, K, device=device)
    w = onehots[:, :1, :].squeeze(1)
    points, phis = [], []
    real_points = []
    transformer_ins =[]
    for t in range(T):
        x_t = data[:, t:t+1, :]
        import pdb
        pdb.set_trace()
        outputs,transformer_ins = model(
            x_t,
            onehots,
            w, kappa,
            state1, state2, state3,
            t,
            transformer_ins,
            x_r=x_r,
            bias=bias,
            type='inference',
        )
        import pdb
        pdb.set_trace()
        end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, state1, state2, state3, phi = outputs
        
        real_end = data[:, t, 0].item()
        real_dx  = data[:, t, 1].item()
        real_dy  = data[:, t, 2].item()
        real_points.append( [real_end, real_dx, real_dy] )

        squeeze_and_detach = lambda x_t: x_t.squeeze().cpu().detach()
        end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho = tuple(map(squeeze_and_detach, [end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho]))

        #bernoulli sample
        sample_end = int(end > 0.5)

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
    real_points = np.array(real_points)
    points = np.array(points)
    plot_stroke(points, save_name=save_name)
    plot_stroke(real_points, save_name=save_name)

    return points
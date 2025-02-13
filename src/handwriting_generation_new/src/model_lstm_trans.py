import math
import numpy as np
import logging
from itertools import chain

# import pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
# device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

# attention window for handwriting synthesis
class Window(nn.Module):
    def __init__(self, cell_size, K):
        super(Window, self).__init__()
        self.linear = nn.Linear(cell_size, 3*K)
        
    def forward(self, x, kappa_old, onehots):
        max_text_len = onehots.shape[1] if len(onehots.shape) == 3 else onehots.shape[0]
        
        params = self.linear(x).exp()
        
        alpha, beta, pre_kappa = params.chunk(3, dim=-1)
        kappa = kappa_old + pre_kappa
        
        indices = torch.from_numpy(np.array(range(max_text_len))).type(torch.FloatTensor).to(x.device)
        gravity = -beta.unsqueeze(2)*(kappa.unsqueeze(2).repeat(1, 1, max_text_len)-indices)**2
        phi = (alpha.unsqueeze(2) * gravity.exp()).sum(dim=1)#*(max_text_len/text_lens)
        
        w = (phi.unsqueeze(2) * onehots).sum(dim=1) 
        return w, kappa, phi

class LSTM1(nn.Module):
    def __init__(self, vocab_len, cell_size, K):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTMCell(input_size = 3 + vocab_len, hidden_size = cell_size)
        self.window = Window(cell_size, K)
        
    def forward(self, x, onehots, w, kappa, state):
        h1s, ws = [], []
        for i in range(x.shape[1]):
            cell_input = torch.cat([x[:, i:i+1].squeeze(1),w], dim=-1)
            state = self.lstm(cell_input, state)
            
            # attention window parameters
            w, kappa, phi = self.window(state[0], kappa, onehots)
            
            # concatenate for single pass through the next layer
            h1s.append(state[0])
            ws.append(w)
        
        return torch.stack(ws, dim=0).permute(1,0,2), torch.stack(h1s, dim=0).permute(1,0,2), state, w, kappa, phi 
    
class Blur1d(nn.Module):
    def __init__(self, kernel, channels, stride):
        super().__init__()
        
        kernel = torch.Tensor(kernel)
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1)

        self.kernel = nn.Parameter(kernel)
        self.channels = channels
        self.stride = stride
        
    def forward(self, X):
        blurred = F.conv1d(X, self.kernel, stride=self.stride, groups=self.channels)
        return blurred


class StyleEncoder(nn.Module):
    def __init__(self, hidden_size, vocab_len, z_size, conv_hidden_size=32, se_hidden_size=128, mha_hidden_size=256, mha_num_head=4):
        super().__init__()

        self.style_featurizer = nn.Sequential(
            Blur1d([1, 3, 3, 1], 3, 1),
            nn.Conv1d(3, conv_hidden_size, 3, stride=2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            Blur1d([1, 3, 3, 1], conv_hidden_size, 1),
            nn.Conv1d(conv_hidden_size, conv_hidden_size * 2, 3, stride=2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            Blur1d([1, 3, 3, 1], conv_hidden_size * 2, 1),
            nn.Conv1d(conv_hidden_size * 2, conv_hidden_size * 4, 3, stride=2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            Blur1d([1, 3, 3, 1], conv_hidden_size * 4, 1),
            nn.Conv1d(conv_hidden_size * 4, conv_hidden_size * 8, 3, stride=2),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        self.query_proj = nn.Linear(hidden_size + vocab_len, mha_hidden_size)
        self.A = nn.Parameter(torch.randn(se_hidden_size, conv_hidden_size * 8), requires_grad=True)
        self.mha = nn.MultiheadAttention(mha_hidden_size, mha_num_head, batch_first=True)
        self.style_dist_proj = nn.Linear(conv_hidden_size * 8, z_size * 2)
        self.prior_dist_proj = nn.Sequential(nn.Linear(hidden_size + vocab_len, hidden_size), nn.ReLU(), nn.Linear(hidden_size, z_size * 2))
        
        self.z_size = z_size
        
    def featurize(self, X):
        return self.style_featurizer(X.permute(0, 2, 1)).permute(0, 2, 1)
    
    def prior(self, h1s, ws):
        prior_dist_params = self.prior_dist_proj(torch.cat([h1s, ws], dim=-1))
        mu, log_var = prior_dist_params[:, :, :self.z_size], prior_dist_params[:, :, self.z_size:]
        
        return mu, log_var
    
    def compute_feats_masks(self, masks):
        feat_masks = torch.clone(masks)
        
        feat_masks[feat_masks == 0] = torch.inf
        for _ in range(4):
            feat_masks = F.max_pool1d(feat_masks, 4, stride=1, padding=0)
            feat_masks = F.max_pool1d(feat_masks, 3, stride=2, padding=0)
        feat_masks[feat_masks == torch.inf] = 0

        return feat_masks
    
    def compute_style_latent(self, feats, masks):
        feats_masks = self.compute_feats_masks(masks)
        style = torch.sum(torch.matmul(feats, self.A.T) * feats_masks.unsqueeze(-1), dim=1) / feats_masks.sum(-1, keepdim=True)

        return style, feats_masks
    
    def forward(self, h1s, ws, feats_r, feats_a=None, masks_r=None, masks_a=None):
        q = F.relu(self.query_proj(torch.cat([h1s, ws], dim=-1)))
        if feats_a is not None:
            q = q * masks_a.unsqueeze(-1)

            self.A.data = F.normalize(self.A, dim=0)
            
            (style_a, _), (style_r, feats_r_masks) = self.compute_style_latent(feats_a, masks_a), self.compute_style_latent(feats_r, masks_r)
            style_transform = torch.matmul(style_a - style_r, self.A).unsqueeze(1).repeat(1, feats_r.shape[1], 1)
            feats_r = (feats_r + style_transform) * feats_r_masks.unsqueeze(-1)
        
        attn_out = self.mha(q, feats_r, feats_r, need_weights=False)[0]
        style_dist_params = self.style_dist_proj(attn_out)
        
        mu, log_var = style_dist_params[:, :, :self.z_size], style_dist_params[:, :, self.z_size:]
        eps = torch.randn_like(log_var)
        z = mu + torch.exp(log_var/2) * eps
                
        return mu, log_var, self.A.T, z

# 2-layer lstm with mixture of gaussian parameters as outputs
# with skip connections
class LSTMSynthesis(nn.Module):
    def __init__(self, vocab_len, cell_size, num_clusters, K, z_size=0):
        super(LSTMSynthesis, self).__init__()
        self.lstm1 = LSTM1(vocab_len, cell_size, K)
        self.lstm2 = nn.LSTM(input_size=3+vocab_len+cell_size+z_size, hidden_size=cell_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=3+vocab_len+cell_size+z_size, hidden_size=cell_size, num_layers=1, batch_first=True)

        self.positional_encoding = PositionalEncoding(cell_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cell_size, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.linear = nn.Linear(cell_size*4, 2+num_clusters*6)
        
        if z_size != 0:
            self.style_encoder = StyleEncoder(cell_size, vocab_len, z_size)
        
    def forward(self, x, onehots, w, kappa, state1, state2, state3, x_r=None, masks=None, masks_r=None, bias=0.):
        ws, h1s, state1, w, kappa, phi = self.lstm1(x, onehots, w, kappa, state1)
        if x_r is not None:
            feats_r, feats_a = self.style_encoder.featurize(x_r), None if masks is None else self.style_encoder.featurize(x)
            mu_z, log_var_z, A, z = self.style_encoder(h1s, ws, feats_r, feats_a=feats_a, masks_r=masks_r, masks_a=masks)
            mu_prior, log_var_prior = self.style_encoder.prior(h1s, ws)
            if torch.isnan(z).any():
                logger.warning('z is nan')
            
        h2s, state2 = self.lstm2(torch.cat([x,ws,h1s] + ([z] if x_r is not None else []), -1), state2)
        h3s, state3 = self.lstm3(torch.cat([x,ws,h2s] + ([z] if x_r is not None else []), -1), state3)        
        
        transformer_in = self.positional_encoding(h3s)
        tgt_mask = self.generate_square_subsequent_mask(transformer_in.size(1)).to(transformer_in.device)
        transformer_out = self.transformer_decoder(transformer_in,tgt_mask=tgt_mask)



        params = self.linear(torch.cat([h1s,h2s,h3s,transformer_out], dim=-1))
        mog_params = params[:, :, :params.shape[-1]-2]
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights*(1+bias), dim=-1)
        rho = F.tanh(pre_rho)
        end = F.sigmoid(params[:, :, params.shape[-1]-2:params.shape[-1]-1])
        stop = F.sigmoid(params[:, :, params.shape[-1]-1:])
        
        for n, a in zip(['ws', 'h1s', 'state10', 'state11', 'w', 'kappa', 'phi', 'h2s', 'state20', 'state21', 'h3s', 'state30', 'state31', 'params'], [ws, h1s, state1[0], state1[1], w, kappa, phi, h2s, state2[0], state2[1], h3s, state3[0], state3[1], params]):
            if torch.isnan(a).any():
                logger.warning(f'{n} is nan')
            if torch.isinf(a).any():
                logger.warning(f'{n} is inf')
                
        ret = (end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, state1, state2, state3, phi)
        if masks is not None:
            ret += (mu_z, log_var_z, mu_prior, log_var_prior, A)
        return ret
    
    def clip_grad_ensure_model_health(self, clip_value, lstm_clip_value):
        for n, p in self.named_parameters():
            if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                logger.warning(f'{n} GRAD is inf or Nan')
                torch.nan_to_num_(p.grad)
            if torch.isnan(p).any() or torch.isinf(p).any():
                logger.warning(f'{n} PARAM is nan or inf')
                torch.nan_to_num_(p)

        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value)
        torch.nn.utils.clip_grad_value_(chain(self.lstm1.parameters(), self.lstm2.parameters(), self.lstm3.parameters()), lstm_clip_value)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

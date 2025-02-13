import math
import numpy as np
import logging
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位置: sin
        # 奇数位置: cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (max_len, d_model) => (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不作为可训练参数

    def forward(self, x):
        """
        x: (B, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 直接相加
        x = x + self.pe[:, :seq_len, :]
        return x



def generate_subsequent_mask(seq_len: int, device=None):
    """
    生成一个针对 Decoder 的下三角 mask。
    上三角部分(未来时刻)设为 -inf,保证自回归时只看当前和过去。
    形状: (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class TransformerDecoderLayers(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=2048, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,   # 让输入维度 (B, seq_len, d_model)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory=None, tgt_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: (B, tgt_len, d_model)  目标序列
        memory: (B, mem_len, d_model) or None
        tgt_mask: (tgt_len, tgt_len)
        """
        if memory is None:
            B, tgt_len, d_model = tgt.size()
            memory = torch.zeros(B, 1, d_model, device=tgt.device)

        out = self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return out  # (B, tgt_len, d_model)


class TransformerSynthesis(nn.Module):
    """
    整体结构:
      - LSTM1 + Window (第1层)
      - (可选) StyleEncoder (此处不展开，可自行保留/删除)
      - 2层 TransformerDecoder
      - 最终 MDN 输出层
    """
    def __init__(self, vocab_len, cell_size, num_clusters, K, z_size=0, use_posenc=True):
        super().__init__()
        self.vocab_len = vocab_len
        self.cell_size = cell_size
        self.num_clusters = num_clusters
        self.K = K
        self.z_size = z_size
        
        # ---------------- 第1层：LSTM+Window ----------------
        self.lstm1 = LSTM1(vocab_len, cell_size, K)
        
        # ---------------- StyleEncoder (可选) ----------------
        if z_size > 0:
            self.style_encoder = StyleEncoder(cell_size, vocab_len, z_size)
        else:
            self.style_encoder = None

        # ---------------- TransformerDecoder 两层 ----------------
        # 输入给 decoder 的维度 =  (3 + vocab_len + cell_size (+ z_size?))
        d_model = self.cell_size
        in_dim = 3 + vocab_len + cell_size + (z_size if z_size>0 else 0)
        
        self.proj1 = nn.Linear(in_dim, d_model)
        self.proj2 = nn.Linear(in_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(in_dim) if use_posenc else None

        # Decoder
        self.transformer1 = TransformerDecoderLayers(
            d_model = in_dim,
            nhead = 4,
            dim_feedforward = 2048,
            num_layers = 1
        )

        self.transformer2 = TransformerDecoderLayers(
            d_model = in_dim,
            nhead = 4,
            dim_feedforward = 2048,
            num_layers = 1
        )
        
        # ---------------- 输出层 (MDN) ----------------
        # 拼接 [h1s, transformer_out] => [cell_size + d_model]
        # 最终输出 2 + num_clusters*6
        self.linear = nn.Linear(d_model*3, 2 + num_clusters*6)

    def forward(self, 
                x,            # (B, seq_len, 3)
                onehots,      # (B, max_text_len, vocab_len)
                w,            # (B, vocab_len)    初始窗口
                kappa,        # (B, K)            初始 kappa
                state1,       # (h1, c1)  => (B, cell_size)
                state2=None,  # 占位
                state3=None,  # 占位
                x_r=None,     # Style 用
                masks=None, 
                masks_r=None,  
                bias=0.0):
        
        # =========== 第1层：LSTM + Window ==============
        ws, h1s, state1, w, kappa, phi = self.lstm1(x, onehots, w, kappa, state1)

        # =========== (可选) Style encoder =============
        if self.style_encoder is not None and x_r is not None:
            feats_r = self.style_encoder.featurize(x_r)          # (B, seq_len2, ?)
            feats_a = None if masks is None else self.style_encoder.featurize(x)
            mu_z, log_var_z, A, z = self.style_encoder(h1s, ws, feats_r, feats_a=feats_a, 
                                                       masks_r=masks_r, masks_a=masks)
            mu_prior, log_var_prior = self.style_encoder.prior(h1s, ws)
        else:
            z = None

        # =========== 组装输入给 Transformer Decoder =============
        if z is not None:
            cat_in = torch.cat([x, ws, h1s, z], dim=-1)  # (B, seq_len, in_dim)
        else:
            cat_in = torch.cat([x, ws, h1s], dim=-1)     # (B, seq_len, in_dim)

        # 位置编码
        if self.pos_encoding is not None:
            transformer_in = self.pos_encoding(cat_in)

        # 构造自回归 mask
        seq_len = transformer_in.size(1)
        tgt_mask = generate_subsequent_mask(seq_len, device=transformer_in.device)
        # 经过 Transformer Decoder
        h2s = self.transformer1(
            tgt=transformer_in, 
            memory=None,        # 如果需要 cross-attention，可传入 encoder 输出
            tgt_mask=tgt_mask
        )  # => (B, seq_len, d_model)

        h2s = self.proj1(h2s)

        h3s = self.transformer2(
            tgt=torch.cat([x,ws,h2s], dim=-1),
            memory=None,
            tgt_mask=tgt_mask
        )

        h3s = self.proj2(h3s)
        # =========== MDN 输出层 ============
        # 拼接 h1s (B, seq_len, cell_size) 和 transformer_out (B, seq_len, d_model)
        final_cat = torch.cat([h1s, h2s,h3s], dim=-1)  # => (B, seq_len, cell_size + d_model)

        # 映射到 MDN 参数: (B, seq_len, 2 + num_clusters*6)
        params = self.linear(final_cat)

        # 分拆 => [pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho] + [end, stop]
        mog_params = params[:, :, :params.shape[-1]-2]
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights*(1+bias), dim=-1)
        rho = F.tanh(pre_rho)
        end = F.sigmoid(params[:, :, params.shape[-1]-2:params.shape[-1]-1])
        stop = F.sigmoid(params[:, :, params.shape[-1]-1:])

        # 数值检查（可选）
        for n, a in zip(['ws','h1s','w','kappa','phi','params', 'h2s', 'h3s'],
                        [ws,     h1s,   w,   kappa,  phi,   params, h2s, h3s]):
            if torch.isnan(a).any():
                logger.warning(f"{n} is nan")
            if torch.isinf(a).any():
                logger.warning(f"{n} is inf")


        ret = (end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, 
               w, kappa, state1, None, None, phi)

        if z is not None:
            ret += (mu_z, log_var_z, mu_prior, log_var_prior, A)
        return ret

    def clip_grad_ensure_model_health(self, clip_value, lstm_clip_value):
        for n, p in self.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    logger.warning(f"{n} grad is inf or nan -> clamp to finite")
                    p.grad = torch.nan_to_num(p.grad)
            if torch.isnan(p).any() or torch.isinf(p).any():
                logger.warning(f"{n} param is inf or nan -> clamp to finite")
                with torch.no_grad():
                    p.copy_(torch.nan_to_num(p))

        # 对全部参数做一次 clip
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value)

        # 对 LSTM1 的参数再额外 clip
        torch.nn.utils.clip_grad_value_(
            chain(self.lstm1.parameters()),
            lstm_clip_value
        )

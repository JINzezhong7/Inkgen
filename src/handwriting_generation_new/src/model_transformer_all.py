import math
import numpy as np
import logging
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
        phi = (alpha.unsqueeze(2) * gravity.exp()).sum(dim=1)  # *(max_text_len/text_lens)
        
        w = (phi.unsqueeze(2) * onehots).sum(dim=1)
        return w, kappa, phi

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
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
      - Transformer + Window (第1层)
      - TransformerDecoder (第2层)
      - 最终 MDN 输出层
    """
    def __init__(self, vocab_len, cell_size, num_clusters, K, z_size=0, use_posenc=True):
        super().__init__()
        self.vocab_len = vocab_len
        self.cell_size = cell_size
        self.num_clusters = num_clusters
        self.K = K
        self.z_size = z_size
        
        # ---------------- 第一层：Transformer + Window ----------------
        self.transformer1 = TransformerDecoderLayers(d_model=cell_size, nhead=4, num_layers=2)
        self.window = Window(cell_size, K)  # 保留原有 Window 计算
        
        # ---------------- TransformerDecoder 第二层 ----------------
        in_dim = 3 + vocab_len + cell_size + (z_size if z_size > 0 else 0)
        self.transformer_proj = nn.Linear(in_dim, cell_size)
        self.pos_encoding = PositionalEncoding(cell_size) if use_posenc else None
        self.transformer2 = TransformerDecoderLayers(d_model=cell_size, nhead=4, num_layers=2)
        
        # 输出层
        self.linear = nn.Linear(cell_size + vocab_len, 2 + num_clusters * 6)

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
        
        # =========== 第1层：Transformer + Window ==============
        # Transformer 输入处理
        transformer_in = x
        transformer_in = self.transformer_proj(transformer_in)  # 投影到 d_model
        if self.pos_encoding is not None:
            transformer_in = self.pos_encoding(transformer_in)  # 加上位置编码
        
        transformer_in = self.transformer1(transformer_in)  # Transformer 处理序列
        
        # Window 计算
        w, kappa, phi = self.window(transformer_in, kappa, onehots)

        # =========== (可选) Style encoder ==============
        if self.style_encoder is not None and x_r is not None:
            feats_r = self.style_encoder.featurize(x_r)
            feats_a = None if masks is None else self.style_encoder.featurize(x)
            mu_z, log_var_z, A, z = self.style_encoder(transformer_in, w, feats_r, feats_a=feats_a, masks_r=masks_r, masks_a=masks)
            mu_prior, log_var_prior = self.style_encoder.prior(transformer_in, w)
        else:
            z = None

        # =========== 组装输入给 Transformer Decoder ==============
        if z is not None:
            cat_in = torch.cat([x, w, transformer_in, z], dim=-1)  # (B, seq_len, in_dim)
        else:
            cat_in = torch.cat([x, w, transformer_in], dim=-1)     # (B, seq_len, in_dim)

        transformer_in = self.transformer_proj(cat_in)

        # 位置编码
        if self.pos_encoding is not None:
            transformer_in = self.pos_encoding(transformer_in)

        # 构造自回归 mask
        seq_len = transformer_in.size(1)
        tgt_mask = generate_subsequent_mask(seq_len, device=transformer_in.device)
        transformer_out = self.transformer2(transformer_in, tgt_mask=tgt_mask)

        # =========== MDN 输出层 ================
        final_cat = torch.cat([transformer_in, transformer_out], dim=-1)  # (B, seq_len, 2*d_model)
        params = self.linear(final_cat)  # (B, seq_len, 2 + num_clusters*6)

        # 拆分 -> [pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho] + [end, stop]
        mog_params = params[..., :params.shape[-1]-2]
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights*(1+bias), dim=-1)
        rho = torch.tanh(pre_rho)

        end = torch.sigmoid(params[..., -2:-1])
        stop = torch.sigmoid(params[..., -1:])
        
        # 返回
        ret = (end, stop, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, state1, None, None, phi)

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ConvNetXtBlock(nn.Module):
    def __init__(
            self,
            kernel_size_pool=3,
            stride_pool=1,
            in_channels=128,
            out_channels=64,
            kernel_size_conv=7,
            stride_conv=2,
            embedding_dim_1=None,
            embedding_dim_2=None,
            in_features=None,
            out_features=None,
            dropout=0.1
        ):
    
        super().__init__()

        embedding_dim_1 = embedding_dim_1 or out_channels
        embedding_dim_2 = embedding_dim_2 or out_channels
        in_features = in_features or out_channels
        out_features = out_features or out_channels * 4

        
        self.pool = (
            nn.AvgPool1d(kernel_size_pool, stride_pool)
            if kernel_size_pool is not None
            else nn.Identity()
        )
    
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_conv, stride=stride_conv, padding=kernel_size_conv // 2, groups=out_channels)
        self.norm1 = nn.LayerNorm(embedding_dim_1)
        self.ffn    = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, in_features),
        )
        self.norm2 = nn.LayerNorm(embedding_dim_2)


        needs_proj = (stride_conv != 1) or (in_channels != out_channels)
        self.skip_proj = (
            nn.Sequential(
                self.pool,                                 # same pool
                nn.Conv1d(in_channels, out_channels, 1,    # 1×1 conv
                          stride=stride_conv)
            ) if needs_proj else nn.Identity()
        )
        


    def forward(self, x):
        skip = self.skip_proj(x)
        
        x = self.pool(x)
        x = self.conv(x)
        x = x.transpose(1,2)   #    Transposing the 1st and 2nd dim (Needed to maintain dim uniformity)
        x = self.norm1(x)
        x = self.ffn(x)
        output = self.norm2(x + skip.transpose(1,2))     # Initially x has 640 cols, so res also must have 640 cols/features

        return output.transpose(1,2)  #     Back to the original dimension





class ConvNetXtEncoder(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        #   128 -> 64
        self.convnet_block_1a = ConvNetXtBlock(stride_conv=1, in_channels=128, out_channels=64, dropout=dropout, kernel_size_pool=None)
        self.convnet_block_1b = ConvNetXtBlock(stride_conv=2, in_channels=64, out_channels=64, dropout=dropout)

        #   64 -> 32
        self.convnet_block_2a = ConvNetXtBlock(stride_conv=1, in_channels=64, out_channels=32, dropout=dropout, kernel_size_pool=None)
        self.convnet_block_2b = ConvNetXtBlock(stride_conv=1, in_channels=32, out_channels=32, dropout=dropout, kernel_size_pool=None)
        self.convnet_block_2c = ConvNetXtBlock(stride_conv=2, in_channels=32, out_channels=32, dropout=dropout)

        #   32 -> 1
        self.convnet_block_3 = ConvNetXtBlock(stride_conv=1, in_channels=32, out_channels=32, dropout=dropout, kernel_size_pool=None)

        self.gap = nn.AdaptiveAvgPool1d(1)    # Does (B, C, 1)

        #   For regression
        self.reg_head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        #   For classification
        self.clas_head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)  #    Need to cross-verify (paper dont have mention of this)
        )


    def forward(self, x):
        #   First stage
        res_1 = self.convnet_block_1a(x)
        res_1 = self.convnet_block_1b(res_1)

        #   Second stage
        res_2 = self.convnet_block_2a(res_1)
        res_2 = self.convnet_block_2b(res_2)
        res_2 = self.convnet_block_2c(res_2)

        #   Third stage
        outputs = self.convnet_block_3(res_2)


        fin_op = self.gap(outputs)  #   (B, 32, 1)
        fin_op = fin_op.flatten(1)  #   (B, 32)

        y_reg = self.reg_head(fin_op)
        y_cls = self.clas_head(fin_op)

        return y_reg.squeeze(-1), y_cls






#   The architecture is correct, but dimension (in-features, out-features etc) need to be adjusted.....!!

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            d_model=128,
            nhead=4,
            dim_feedforward=128*4,
            dropout=0.1,
        ):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)

    
    def forward(self, x, mask=None):
        return self.layer(x, src_key_padding_mask=mask)
    


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            in_dim=640,          # dimension of incoming features per token
            num_layers=4,
            d_model=128,
            nhead=4,
            dim_feedforward=128*4,
            dropout=0.1,
            max_len=21
        ):
        super().__init__()


        self.in_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.encoder_layers = nn.ModuleList()
        
        self.encoder_layers.extend([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)


    def forward(self, x, mask=None):
        x = self.in_proj(x)           # (B, L, d_model)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.norm(x)              # (B, L, d_model)
        return x
    



class DeepSilencer(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 4, nhead: int = 4,
                 dim_ff: int = 128 * 4, dropout: float = 0.1,
                 in_dim: int = 640, tr_dim: int | None = None):
        super().__init__()
        self.transformer = TransformerEncoder(
            in_dim=in_dim,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout
        )

        self.tr_proj = None
        if tr_dim is not None and tr_dim > 0:
            self.tr_proj = nn.Sequential(
                nn.LayerNorm(tr_dim),
                nn.Linear(tr_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
       
        self.convnext = ConvNetXtEncoder(dropout=dropout)


    def forward_once(self, x, tr=None):
        # x: (B, L, in_dim)
        h = self.transformer(x)  # (B, L, d_model)
        
        if self.tr_proj is not None and tr is not None:
            b = self.tr_proj(tr)           # (B, d_model)
            h = h + b.unsqueeze(1)         # add as bias to each token
        h = h.transpose(1, 2)              # (B, d_model, L)
        
        y_reg, y_cls = self.convnext(h)
        return y_reg, y_cls
    

    def forward(self, e1, e2, tr1=None, tr2=None):
        y1_reg, y1_cls = self.forward_once(e1, tr1)
        y2_reg, y2_cls = self.forward_once(e2, tr2)
        return (y1_reg, y1_cls, y2_reg, y2_cls)
    
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """标准正余弦位置编码 """

    def __init__(self, d_model: int, max_len: int = 1000 + 1):
        super().__init__()
        # 预先计算位置编码矩阵，注册为缓冲区避免参与梯度计算
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor):
        """x: (B, L, D)"""
        L = x.size(1)
        return x + self.pe[:, :L]


class SpectraTransformer(nn.Module):
    """基于 Transformer Encoder 的 Raman 光谱分类模型."""

    def __init__(
        self,
        input_dim: int = 1000,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_classes: int = 30,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_classes = n_classes
        self.use_cls_token = use_cls_token

        # 将每个采样点 (1维) 投影到 d_model
        self.token_embed = nn.Linear(1, d_model)

        if self.use_cls_token:
            # 可学习的 [CLS] token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用 (B, L, D)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.token_embed.weight)
        if self.token_embed.bias is not None:
            nn.init.zeros_(self.token_embed.bias)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

    def encode(self, x: torch.Tensor):
        """
        x: (B, 1, input_dim)  ->  返回 (B, d_model)
        """
        # 调整形状 (B, input_dim, 1)
        x = x.permute(0, 2, 1)
        # Token embedding -> (B, L, D)
        x = self.token_embed(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B,1,D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, L+1, D)

        # 加入位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.encoder(x)

        # 取表示
        if self.use_cls_token:
            z = x[:, 0]  # (B, D)
        else:
            z = x.mean(dim=1)  # (B, D)
        z = self.norm(z)
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.fc(z) 
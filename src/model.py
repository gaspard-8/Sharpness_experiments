import torch

from torch import nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_attn_heads: int,
        num_layers: int,
        vocab_size: int = 2,
        dropout: float = 0.0,
        max_len: int = 1024,
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even because embeddings are concatenated.")
        if d_model % num_attn_heads != 0:
            raise ValueError("d_model must be divisible by num_attn_heads.")

        self.embedding = nn.Embedding(vocab_size, d_model // 2)
        self.positional_encoding = nn.Embedding(max_len, d_model // 2)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_attn_heads,
                dropout=dropout,
                activation="relu",
                dim_feedforward=d_model,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.linear_head = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = self.positional_encoding(
            torch.arange(seq_len, device=x.device)
        ).unsqueeze(0)
        positions = positions.expand(x.size(0), -1, -1)
        x = torch.cat((self.embedding(x), positions), dim=-1)
        x = self.encoder(x)
        return self.linear_head(x)[:, -1, 0]

from typing import Optional, Dict, Any, Tuple
import numpy as np

from config import CFG
from utils.logger import log


class TemporalFusionTransformer:
    def __init__(self):
        self._model = None
        self._has_torch = False
        self._check_frameworks()

    def _check_frameworks(self):
        try:
            import torch
            self._has_torch = True
        except ImportError:
            pass

    def build(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> Optional[Any]:
        if not self._has_torch:
            log.warning("PyTorch not available for TFT")
            return None

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class _TFTBlock(nn.Module):
                def __init__(self, hidden_dim, num_heads, dropout):
                    super().__init__()
                    self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                    self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                    self.ff = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 4, hidden_dim),
                    )
                    self.norm1 = nn.LayerNorm(hidden_dim)
                    self.norm2 = nn.LayerNorm(hidden_dim)
                    self.norm3 = nn.LayerNorm(hidden_dim)
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    attn_out, _ = self.attention(x, x, x)
                    x = self.norm1(x + self.dropout(attn_out))
                    lstm_out, _ = self.lstm(x)
                    x = self.norm2(x + self.dropout(lstm_out))
                    ff_out = self.ff(x)
                    x = self.norm3(x + self.dropout(ff_out))
                    return x

            class _TFT(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    self.blocks = nn.ModuleList([
                        _TFTBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
                    ])
                    self.output = nn.Linear(hidden_dim, 1)

                def forward(self, x):
                    x = self.input_proj(x)
                    for block in self.blocks:
                        x = block(x)
                    return self.output(x[:, -1, :])

            self._model = _TFT(input_dim, hidden_dim, num_heads, num_layers, dropout)
            return self._model

        except Exception as e:
            log.warning(f"TFT build failed: {e}")
            return None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int = 30,
        **kwargs,
    ) -> Optional[Any]:
        if not self._has_torch:
            return None

        try:
            import torch
            import torch.nn as nn

            if self._model is None:
                _, features = X_train.shape
                self.build(features)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)

            seqs = X_train.shape[0] // sequence_length
            X_seq = X_train[:seqs * sequence_length].reshape(seqs, sequence_length, -1)
            y_seq = y_train[:seqs * sequence_length].reshape(seqs, sequence_length)[:, -1]

            X_t = torch.FloatTensor(X_seq).to(device)
            y_t = torch.FloatTensor(y_seq).to(device)

            optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            epochs = kwargs.get("epochs", min(50, CFG.ml.epochs))

            for epoch in range(epochs):
                self._model.train()
                optimizer.zero_grad()
                output = self._model(X_t).squeeze()
                loss = criterion(output, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            return self._model

        except Exception as e:
            log.warning(f"TFT training failed: {e}")
            return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(X.shape[0])
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                if len(X.shape) == 2:
                    X_t = X_t.unsqueeze(1)
                pred = self._model(X_t).cpu().numpy().flatten()
            return pred
        except Exception:
            return np.zeros(X.shape[0])

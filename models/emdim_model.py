import torch
import torch.nn as nn

class EMDIM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.classifier_head = nn.Linear(emb_dim, 2)
        self.risk_head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.mutation_head = nn.Linear(emb_dim, 1)  # Optional, keep if you want mutation map

    def forward(self, x):
        x = self.embedding(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        
        class_logits = self.classifier_head(pooled)
        risk_score = self.risk_head(pooled).squeeze(1)
        
        # Optional: mutation output (position-wise)
        mutation_map = self.mutation_head(encoded).squeeze(-1)
        
        return class_logits, mutation_map, risk_score

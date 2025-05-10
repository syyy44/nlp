import math
import torch.nn as nn
import torch
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len,1,d_model)
    def forward(self, x):  # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]

class TransformerMT(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_enc=6, num_dec=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_enc, num_dec, dim_ff, dropout, batch_first=True)
        self.fc_out      = nn.Linear(d_model, vocab_size)
        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, 
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # src/tgt: (S,B) & (T,B)
        src_emb = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb,
                               src_mask=src_mask, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(out)  # (T,B,vocab_size)

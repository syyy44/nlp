import torch.optim as optim
import torch.nn as nn
import torch
from model import TransformerMT
from data_loader import CustomDataset, collate_fn
from torch.utils.data import DataLoader
N_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomDataset('data/train.txt','\t')
loader  = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
model = TransformerMT(vocab_size=len(dataset.word2id_src), d_model=512).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2id_src['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.98), eps=1e-9)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    return mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

for epoch in range(1, N_epochs+1):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1]
        src_mask, tgt_mask = None, generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask=(src==dataset.word2id_src['<PAD>']).T,
                       tgt_padding_mask=(tgt_input==dataset.word2id_src['<PAD>']).T,
                       memory_key_padding_mask=(src==dataset.word2id_src['<PAD>']).T)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

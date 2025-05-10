import torch
from torch.utils.data import Dataset
from data_process import word_to_id
class CustomDataset(Dataset):
    def __init__(self, pth, sp = '\t'):
        self.pth = pth
        self.word2id_src = word_to_id('data/word2int_en.json')
        self.word2id_tgt = word_to_id('data/word2int_cn.json')
        self.sp = sp
        self.pairs = []
        with open(pth, 'r') as f:
            for line in f:
                src_ids, tgt_ids = line.strip().split(sp)
                src_ids = [int(id) for id in src_ids.split()]
                tgt_ids = [1] + [int(id) for id in tgt_ids.split()] + [2]
                self.pairs.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return src_ids, tgt_ids
    
def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src, max_tgt = max(src_lens), max(tgt_lens)
    pad = 0
    src_pad = [s + [pad]*(max_src-len(s)) for s in srcs]
    tgt_pad = [t + [pad]*(max_tgt-len(t)) for t in tgts]
    return (torch.tensor(src_pad).T, torch.tensor(tgt_pad).T)

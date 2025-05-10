import json
import os

def word_to_id(pth):
    #打开json文件
    with open(pth, 'r') as f:
        word_to_id = json.load(f)
    return word_to_id

def process_data(raw_pth, pth, sp):
    new_data = []
    word2id_cn = word_to_id('data/word2int_cn.json')
    word2id_en = word_to_id('data/word2int_en.json')

    with open(raw_pth, 'r') as f:
        for line in f:
            src, tgt = line.strip().split(sp)
            src_ids = [word2id_en[word] if word in word2id_en else word2id_en['<UNK>'] for word in src.split()]
            tgt_ids = [word2id_cn[word] if word in word2id_cn else word2id_cn['<UNK>'] for word in tgt.split()]
            new_data.append((src_ids, tgt_ids))

    with open(pth, 'w', encoding='utf-8') as f:
        #输出到txt文件
        for src_ids, tgt_ids in new_data:
            f.write(" ".join(str(id) for id in src_ids) + sp + " ".join(str(id) for id in tgt_ids) + '\n')

if __name__ == '__main__':
    sp = '\t'
    process_data('data/training.txt', 'data/train.txt', sp)
    process_data('data/testing.txt', 'data/test.txt', sp)
    process_data('data/validation.txt', 'data/valid.txt', sp)

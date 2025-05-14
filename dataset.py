import spacy
import torch
from torch.utils.data import Dataset

TOTAL_DATA_NUM = 5000

# 加载 spacy 模型
spacy_zh = spacy.load('zh_core_web_sm')
spacy_en = spacy.load('en_core_web_sm')

# 中文和英文的 tokenizer，直接用 lambda 封装 spacy 调用
def tokenize_zh(text):
    return [tok.text for tok in spacy_zh(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en(text)]

class WordDataset(Dataset):
    def __init__(self, src_file, trg_file):
        self.src_lines = open(src_file, encoding='utf-8').readlines()[:TOTAL_DATA_NUM]
        self.trg_lines = open(trg_file, encoding='utf-8').readlines()[:TOTAL_DATA_NUM]

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx].strip()
        trg_line = self.trg_lines[idx].strip()

        src_tokens = tokenize_zh(src_line)
        trg_tokens = tokenize_en(trg_line)

        return src_tokens, trg_tokens

class NumberDataset(Dataset):
    def __init__(self, src_file, trg_file, src_vocab, trg_vocab, max_len):
        self.src_lines = open(src_file, encoding='utf-8').readlines()[:TOTAL_DATA_NUM]
        self.trg_lines = open(trg_file, encoding='utf-8').readlines()[:TOTAL_DATA_NUM]

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx].strip()
        trg_line = self.trg_lines[idx].strip()

        src_tokens = tokenize_zh(src_line)
        trg_tokens = tokenize_en(trg_line)

        # Convert to indices
        src_nums = [self.src_vocab.get(word, None) for word in src_tokens if word in self.src_vocab]
        trg_nums = [self.trg_vocab.get(word, None) for word in trg_tokens if word in self.trg_vocab]

        # 添加 <sos> 和 <eos>，用1和2表示
        src_nums = [1] + src_nums + [2]
        trg_nums = [1] + trg_nums + [2]

        # Padding
        src_res = [0] * self.max_len
        trg_res = [0] * self.max_len
        src_res[:len(src_nums)] = src_nums[:self.max_len]
        trg_res[:len(trg_nums)] = trg_nums[:self.max_len]

        return torch.tensor(src_res), torch.tensor(trg_res)

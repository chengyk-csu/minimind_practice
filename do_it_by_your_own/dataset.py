import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from model import MiniMindConfig

class PretrainDataset(Dataset):
    def __init__(self,train_path,tokenizer,config: MiniMindConfig):
        super().__init__()
        self.samples = load_dataset("json",data_files=train_path,split="train")
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        sample = self.samples[index]
        text = str(sample['text'])
        input_ids = self.tokenizer.encode(
            text=text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len-2)
        #input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        input_ids = input_ids + [self.tokenizer.pad_token_id]*(self.max_seq_len-len(input_ids))
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[labels==self.tokenizer.pad_token_id] = -100
        return input_ids,attention_mask, labels


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = load_dataset("json",data_files=data_path,split="train")
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n',add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n',add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self, conversations):
        return self.tokenizer.apply_chat_template(conversations,tokenize=False,add_generation_prompt=False)
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if (input_ids[end:end + len(self.eos_id)]== self.eos_id):
                        break
                    end += 1
                for j in range(start,min(end + len(self.eos_id),self.max_seq_len)):
                    labels[j] = input_ids[j]
                if end < len(input_ids):
                    i = end + len(self.eos_id)
                else:
                    i = len(input_ids)
            else:
                i += 1
        return labels
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample["conversations"]
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt,add_special_tokens=False).input_ids
        # 修复：构造函数保存的字段名是 max_seq_len。
        input_ids = input_ids[:self.max_seq_len]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))
        labels = self.generate_labels(input_ids)
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        labels = torch.tensor(labels,dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, attention_mask, labels

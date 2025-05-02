from torch.utils.data import Dataset
import torch
import pickle
from torch.utils.data import random_split

pkl_path = "processed_dataset.pkl"

# class ProteinDataset(Dataset):
#     def __init__(self, df):
#         with open(pkl_path, "rb") as f:
#             self.samples = pickle.load(f)
        
#         #self.samples = li
        
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]
    
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }
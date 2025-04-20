from torch.utils.data import Dataset
import pickle
from transformers import EsmForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import random_split

pkl_path = "C:\\Users\\lovem\\Desktop\\repository\\biologyInformation\\protein_ml_project\\processed_dataset.pkl"

class ProteinDataset(Dataset):
    def __init__(self, df):
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        
        #self.samples = li
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

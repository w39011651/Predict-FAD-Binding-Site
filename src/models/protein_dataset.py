from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, data_list):
        """for li, li should have structure:[{input_ids:...,attention_mask:...,labels:...}]"""
        self.samples = data_list
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            return self.samples[idx]
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return None  # 或 raise，讓你能看到是哪筆資料報錯
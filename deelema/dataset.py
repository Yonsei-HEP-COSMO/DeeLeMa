import torch
from torch.utils.data import Dataset

# Dataset class for the toy example
class ToyData(Dataset):
    def __init__(self, p_A, p_a, p_B, p_b, q_C, q_c):
        self.X = torch.column_stack([p_A, p_a, p_B, p_b])
        self.q_C = q_C
        self.q_c = q_c
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.q_C[idx], self.q_c[idx]

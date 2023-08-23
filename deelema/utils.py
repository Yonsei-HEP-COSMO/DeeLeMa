import torch
import numpy as np

def squared_mass(p):
    return p[:,0]**2 - p[:,1]**2 - p[:,2]**2 - p[:,3]**2

def mass_torch(p, ax=1):
    return torch.sqrt(p[:,0]**2 - p[:,1]**2 - p[:,2]**2 - p[:,3]**2)

def mass_numpy(p, ax=1):
    return np.sqrt(p[:,0]**2 - p[:,1]**2 - p[:,2]**2 - p[:,3]**2)

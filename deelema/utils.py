import torch
import numpy as np

def squared_mass(p):
    return p[:,0]**2 - p[:,1]**2 - p[:,2]**2 - p[:,3]**2

def mass_torch(p, ax=1):
    return torch.sqrt(p[:,0]**2 - p[:,1]**2 - p[:,2]**2 - p[:,3]**2)

def mass_numpy(p, ax=1):
    return np.sqrt(p[:,0]**2 - p[:,1]**2 - p[:,2]**2 - p[:,3]**2)

def q_completion(q, x, m_C):
    """
    From output of the network, complete the 4-momenta of missing particles

    Parameters
    ----------
    q : torch.Tensor
        4-momenta of the observed particles
    x : visible 4-momenta (input data)
    m_C : float
        mass of the missing particle (predetermined)
    """
    qx1 = q[:,0:1]
    qy2 = q[:,1:2]
    qz1 = q[:,2:3]
    qz2 = q[:,3:4]

    pTx = x[:,1:2]+x[:,5:6]+x[:,9:10]+x[:,13:14]
    pTy = x[:,2:3]+x[:,6:7]+x[:,10:11]+x[:,14:15]

    qx2 = -pTx-qx1
    qy1 = -pTy-qy2

    Eq1 = torch.sqrt(m_C**2 + qx1**2 + qy1**2 + qz1**2)
    Eq2 = torch.sqrt(m_C**2 + qx2**2 + qy2**2 + qz2**2)

    qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
    qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)
    
    return torch.column_stack([qc1, qc2])

def decoder(model, dl, m_C):
    """
    Decode the 4-momenta of the missing particles

    Parameters
    ----------
    model : torch.nn.Module
        The trained model
    dl : torch.utils.data.DataLoader
        DataLoader object
    m_C : float
        mass of the missing particle (predetermined)
    """
    model.eval()
    with torch.no_grad():
        dl_iter = iter(dl)
        x_init, _, _ = next(dl_iter)
        output = model(x_init)
        q = q_completion(output, x_init, m_C)
        for data in dl_iter:
            x, _, _ = data
            output = model(x)
            q = torch.concat([q, q_completion(output, x, m_C)])
    return q
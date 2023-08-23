import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
import lightning as pl
from deelema import squared_mass, mass_torch

class DeeLeMa(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        self.hidden_nodes  = hparams["hidden_nodes"]
        self.hidden_layers = hparams["hidden_layers"]
        self.learning_rate = hparams["learning_rate"]
        self.batch_size    = hparams["batch_size"]
        self.epochs        = hparams["epochs"]
        self.learn_mode    = hparams['learn_mode']              # for pT mC loss on off
        self.mass_squared  = hparams['mass_squared']            # True or False
        self.dist_fn       = hparams['dist_fn'].to(self.device) # Determine loss function

        m_C = torch.tensor(hparams["m_C_init"])
        m_B_add = torch.tensor(hparams["m_B_add"])
        m_A_add = torch.tensor(hparams["m_A_add"])

        m_B = m_C + m_B_add
        m_A = m_B + m_A_add

        if self.mass_squared:
            m_C = m_C ** 2
            m_B = m_B ** 2
            m_A = m_A ** 2
        self.m_C = m_C
        self.m_B_add = nn.Parameter(m_B_add, requires_grad=True)
        self.m_A_add = nn.Parameter(m_A_add, requires_grad=True)
        self.m_B = self.m_C + self.m_B_add
        self.m_A = self.m_B + self.m_A_add

        layers = [nn.Linear(16, self.hidden_nodes), nn.GELU(approximate='tanh')]
        for _ in range(self.hidden_layers):
            layers.extend([
                nn.Linear(self.hidden_nodes, self.hidden_nodes),
                nn.GELU(approximate='tanh'),
        ])

        if self.learn_mode == 'pt_mc':
            layers.append(nn.Linear(self.hidden_nodes, 8))
        elif self.learn_mode in ['pt', 'mc']:
            layers.append(nn.Linear(self.hidden_nodes, 6))
        else:
            layers.append(nn.Linear(self.hidden_nodes, 4))

        self.net = nn.Sequential(*layers)
        self.save_hyperparameters(hparams)

    def compute_mass(self):
        self.m_B = self.m_C + F.elu(self.m_B_add) + 1
        self.m_A = self.m_B + F.elu(self.m_A_add) + 1

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, _, _ = batch
        pa1 = x[:,0:4]
        pa2 = x[:,4:8]
        pb1 = x[:,8:12]
        pb2 = x[:,12:16]

        q = self(x)
        if self.learn_mode == 'pt_mc':
            qc1 = q[:,0:4]
            qc2 = q[:,4:8]
        elif self.learn_mode == 'pt':
            qx1 = q[:,0:1]
            qy1 = q[:,1:2]
            qx2 = q[:,2:3]
            qy2 = q[:,3:4]
            qz1 = q[:,4:5]
            qz2 = q[:,5:6]

            Eq1 = torch.sqrt(self.m_C**2 + qx1**2 + qy1**2 + qz1**2)
            Eq2 = torch.sqrt(self.m_C**2 + qx2**2 + qy2**2 + qz2**2)

            qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
            qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)

        elif self.learn_mode == 'mc':
            qx1 = q[:,0:1]
            qy2 = q[:,1:2]
            qz1 = q[:,2:3]
            qz2 = q[:,3:4]
            Eq1 = q[:,4:5]
            Eq2 = q[:,5:6]

            pTx = x[:,1:2]+x[:,5:6]+x[:,9:10]+x[:,13:14]
            pTy = x[:,2:3]+x[:,6:7]+x[:,10:11]+x[:,14:15]

            qx2 = -pTx-qx1
            qy1 = -pTy-qy2

            qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
            qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)

        elif self.learn_mode == None:
            qx1 = q[:,0:1]
            qy2 = q[:,1:2]
            qz1 = q[:,2:3]
            qz2 = q[:,3:4]

            pTx = x[:,1:2]+x[:,5:6]+x[:,9:10]+x[:,13:14]
            pTy = x[:,2:3]+x[:,6:7]+x[:,10:11]+x[:,14:15]

            qx2 = -pTx-qx1
            qy1 = -pTy-qy2

            Eq1 = torch.sqrt(self.m_C**2 + qx1**2 + qy1**2 + qz1**2)
            Eq2 = torch.sqrt(self.m_C**2 + qx2**2 + qy2**2 + qz2**2)

            qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
            qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)

        pB1 = pb1 + qc1
        pB2 = pb2 + qc2
        pA1 = pa1 + pB1
        pA2 = pa2 + pB2
        pT = (pA1 + pA2)[:,1:3]

        if self.mass_squared == True:
            mC1_sq = squared_mass(qc1)
            mC2_sq = squared_mass(qc2)
            mB1_sq = squared_mass(pB1)
            mB2_sq = squared_mass(pB2)
            mA1_sq = squared_mass(pA1)
            mA2_sq = squared_mass(pA2)
        else:
            mC1_sq = mass_torch(qc1)
            mC2_sq = mass_torch(qc2)
            mB1_sq = mass_torch(pB1)
            mB2_sq = mass_torch(pB2)
            mA1_sq = mass_torch(pA1)
            mA2_sq = mass_torch(pA2)

        self.compute_mass()

        mCs = self.m_C * torch.ones_like(mC1_sq)
        mBs = self.m_B * torch.ones_like(mB1_sq)
        mAs = self.m_A * torch.ones_like(mA1_sq)

        metric = self.dist_fn

        loss_C = metric(mC1_sq, mC2_sq) + metric(mC1_sq, mCs) + metric(mC2_sq, mCs)
        loss_B = metric(mB1_sq, mB2_sq) + metric(mB1_sq, mBs) + metric(mB2_sq, mBs)
        loss_A = metric(mA1_sq, mA2_sq) + metric(mA1_sq, mAs) + metric(mA2_sq, mAs)
        loss_pT = pT[:,0]**2 + pT[:,1]**2

        if self.learn_mode == 'pt_mc':
            loss = (loss_A + loss_B + loss_C).mean() + loss_pT.mean()
        elif self.learn_mode == 'pt':
            loss = (loss_A + loss_B).mean() + loss_pT.mean()
        elif self.learn_mode == 'mc':
            loss = (loss_A + loss_B + loss_C).mean()
        else:
            loss = (loss_A + loss_B).mean()
        return loss

    def validation_step(self, batch, _):
        x, _, _ = batch
        pa1 = x[:,0:4]
        pa2 = x[:,4:8]
        pb1 = x[:,8:12]
        pb2 = x[:,12:16]

        q = self(x)
        if self.learn_mode == 'pt_mc':
            qc1 = q[:,0:4]
            qc2 = q[:,4:8]
        elif self.learn_mode == 'pt':
            qx1 = q[:,0:1]
            qy1 = q[:,1:2]
            qx2 = q[:,2:3]
            qy2 = q[:,3:4]
            qz1 = q[:,4:5]
            qz2 = q[:,5:6]

            Eq1 = torch.sqrt(self.m_C**2 + qx1**2 + qy1**2 + qz1**2)
            Eq2 = torch.sqrt(self.m_C**2 + qx2**2 + qy2**2 + qz2**2)

            qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
            qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)

        elif self.learn_mode == 'mc':
            qx1 = q[:,0:1]
            qy2 = q[:,1:2]
            qz1 = q[:,2:3]
            qz2 = q[:,3:4]
            Eq1 = q[:,4:5]
            Eq2 = q[:,5:6]

            pTx = x[:,1:2]+x[:,5:6]+x[:,9:10]+x[:,13:14]
            pTy = x[:,2:3]+x[:,6:7]+x[:,10:11]+x[:,14:15]

            qx2 = -pTx-qx1
            qy1 = -pTy-qy2

            qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
            qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)
        else:
            qx1 = q[:,0:1]
            qy2 = q[:,1:2]
            qz1 = q[:,2:3]
            qz2 = q[:,3:4]

            pTx = x[:,1:2]+x[:,5:6]+x[:,9:10]+x[:,13:14]
            pTy = x[:,2:3]+x[:,6:7]+x[:,10:11]+x[:,14:15]

            qx2 = -pTx-qx1
            qy1 = -pTy-qy2

            Eq1 = torch.sqrt(self.m_C**2 + qx1**2 + qy1**2 + qz1**2)
            Eq2 = torch.sqrt(self.m_C**2 + qx2**2 + qy2**2 + qz2**2)

            qc1  = torch.cat([Eq1,qx1,qy1,qz1], 1)
            qc2  = torch.cat([Eq2,qx2,qy2,qz2], 1)

        pB1 = pb1 + qc1
        pB2 = pb2 + qc2
        pA1 = pa1 + pB1
        pA2 = pa2 + pB2
        pT = (pA1 + pA2)[:,1:3]

        if self.mass_squared == True:
            mC1_sq = squared_mass(qc1)
            mC2_sq = squared_mass(qc2)
            mB1_sq = squared_mass(pB1)
            mB2_sq = squared_mass(pB2)
            mA1_sq = squared_mass(pA1)
            mA2_sq = squared_mass(pA2)
        else:
            mC1_sq = mass_torch(qc1)
            mC2_sq = mass_torch(qc2)
            mB1_sq = mass_torch(pB1)
            mB2_sq = mass_torch(pB2)
            mA1_sq = mass_torch(pA1)
            mA2_sq = mass_torch(pA2)

        self.compute_mass()

        mCs = self.m_C * torch.ones_like(mC1_sq)
        mBs = self.m_B * torch.ones_like(mB1_sq)
        mAs = self.m_A * torch.ones_like(mA1_sq)

        metric = self.dist_fn

        loss_C = metric(mC1_sq, mC2_sq) + metric(mC1_sq, mCs) + metric(mC2_sq, mCs)
        loss_B = metric(mB1_sq, mB2_sq) + metric(mB1_sq, mBs) + metric(mB2_sq, mBs)
        loss_A = metric(mA1_sq, mA2_sq) + metric(mA1_sq, mAs) + metric(mA2_sq, mAs)
        loss_pT = pT[:,0]**2 + pT[:,1]**2

        if self.learn_mode == 'pt_mc':
            loss = (loss_A + loss_B + loss_C).mean() + loss_pT.mean()
        elif self.learn_mode == 'pt':
            loss = (loss_A + loss_B).mean() + loss_pT.mean()
        elif self.learn_mode == 'mc':
            loss = (loss_A + loss_B + loss_C).mean()
        else:
            loss = (loss_A + loss_B).mean()

        self.log('val_loss', loss)
        self.log('loss_A', loss_A.mean())
        self.log('loss_B', loss_B.mean())
        self.log('loss_C', loss_C.mean())
        self.log('loss_pT', loss_pT.mean())
        self.log('m_A', self.m_A)
        self.log('m_B', self.m_B)
        self.log('m_C', self.m_C)
        self.log('m_A1', mA1_sq.mean())
        self.log('m_A2', mA2_sq.mean())
        self.log('m_B1', mB1_sq.mean())
        self.log('m_B2', mB2_sq.mean())
        self.log('m_C1', mC1_sq.mean())
        self.log('m_C2', mC2_sq.mean())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": PolynomialLR(
                    optimizer,
                    total_iters = self.epochs,
                    power = 2.0
                ),
                "interval": "epoch",
                "monitor": "val_loss",
                "strict": True,
            }
        }

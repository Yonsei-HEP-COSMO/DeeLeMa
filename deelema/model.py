import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
import lightning as pl
from deelema import squared_mass, mass_torch

class DeeLeMa(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        # Setting the hyperparameters from the provided input
        self.hidden_nodes  = hparams["hidden_nodes"]
        self.hidden_layers = hparams["hidden_layers"]
        self.learning_rate = hparams["learning_rate"]
        self.batch_size    = hparams["batch_size"]
        self.epochs        = hparams["epochs"]
        self.learn_mode    = hparams['learn_mode']              # Mode for pT mC loss
        self.mass_squared  = hparams['mass_squared']            # Boolean for mass squared
        self.dist_fn       = hparams['dist_fn'].to(self.device) # Loss function
        
        # Initializing masses from the hyperparameters
        m_C = torch.tensor(hparams["m_C_init"])
        m_B_add = torch.tensor(hparams["m_B_add"])
        m_A_add = torch.tensor(hparams["m_A_add"])

        m_B = m_C + m_B_add
        m_A = m_B + m_A_add

        # Check for the squared mass flag and square if true
        if self.mass_squared:
            m_C = m_C ** 2
            m_B = m_B ** 2
            m_A = m_A ** 2
        self.m_C = m_C

        # Convert additional mass terms into trainable parameters
        self.m_B_add = nn.Parameter(m_B_add, requires_grad=True)
        self.m_A_add = nn.Parameter(m_A_add, requires_grad=True)
        self.m_B = self.m_C + self.m_B_add
        self.m_A = self.m_B + self.m_A_add

        # Constructing the neural network architecture
        layers = [nn.Linear(16, self.hidden_nodes), nn.GELU(approximate='tanh')]
        for _ in range(self.hidden_layers):
            layers.extend([
                nn.Linear(self.hidden_nodes, self.hidden_nodes),
                nn.GELU(approximate='tanh'),
        ])

        # Append the last layer based on the learning mode
        if self.learn_mode == 'pt_mc':
            layers.append(nn.Linear(self.hidden_nodes, 8))
        elif self.learn_mode in ['pt', 'mc']:
            layers.append(nn.Linear(self.hidden_nodes, 6))
        else:
            layers.append(nn.Linear(self.hidden_nodes, 4))

        # Construct the neural network
        self.net = nn.Sequential(*layers)

        # Save the hyperparameters
        self.save_hyperparameters(hparams)


    def compute_mass(self):
        """
        Recompute the masses of particles B and A.

        This function updates the masses of particles B and A based on the 
        current values of the trainable parameters m_B_add and m_A_add. 
        These values are added to the existing base mass of particle C to 
        determine the masses of B and A. 

        Note: This function is called to ensure that the masses are 
        updated when the trainable parameters change during training.
        """
        self.m_B = self.m_C + F.elu(self.m_B_add) + 1
        self.m_A = self.m_B + F.elu(self.m_A_add) + 1

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        """
        Computes the forward pass and the loss for a single training batch.

        This function processes the input batch to compute the forward pass based 
        on the current model state. It then calculates the loss based on the 
        discrepancies between predicted quantities and their expected values. 
        The method operates in different modes (`pt_mc`, `pt`, `mc`, or None) 
        which dictate how the forward pass and loss computations are performed. 
        The mode is determined by the attribute `self.learn_mode`.

        Args:
            batch (tuple): Input data batch. It contains tensors with different 
                        quantities, but only a subset of these quantities 
                        is used based on the `learn_mode`.
            _ (int): Placeholder for batch index which is unused in this method.

        Returns:
            torch.Tensor: Computed loss for the current batch which will be 
                        back-propagated to update the model parameters.

        Notes:
            - The method heavily utilizes tensor slicing to extract relevant 
            quantities from the input batch. 
            - It also performs several tensor manipulations to compute quantities 
            for loss calculation.
            - Loss is calculated based on discrepancies between predicted and 
            expected masses of certain particles and also the transverse momentum 
            (pT) in some modes.
        """
        # Extract input features from the given batch.
        x, _, _ = batch

        # Split the input tensor into specific particle attributes.
        pa1, pa2 = x[:, 0:4], x[:, 4:8]
        pb1, pb2 = x[:, 8:12], x[:, 12:16]

        # Forward pass: get model's output based on the input batch.
        q = self(x)

        # Handle different learning modes: 'pt_mc', 'pt', and 'mc'.
        # Each mode modifies how the forward pass output 'q' is used to compute various quantities.
        if self.learn_mode == 'pt_mc':
            qc1, qc2 = q[:, 0:4], q[:, 4:8]
            
        elif self.learn_mode == 'pt':
            # Extract momentum components from the output.
            qx1, qy1, qx2, qy2 = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
            qz1, qz2 = q[:, 4:5], q[:, 5:6]

            # Compute energy for two particles.
            Eq1 = torch.sqrt(self.m_C**2 + qx1**2 + qy1**2 + qz1**2)
            Eq2 = torch.sqrt(self.m_C**2 + qx2**2 + qy2**2 + qz2**2)

            # Concatenate energy and momentum components.
            qc1, qc2 = torch.cat([Eq1, qx1, qy1, qz1], 1), torch.cat([Eq2, qx2, qy2, qz2], 1)
            
        elif self.learn_mode == 'mc':
            # Extract momentum and energy components.
            qx1, qy2, qz1, qz2, Eq1, Eq2 = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4], q[:, 4:5], q[:, 5:6]

            # Calculate total transverse momentum for x and y directions.
            pTx, pTy = x[:, 1:2]+x[:, 5:6]+x[:, 9:10]+x[:, 13:14], x[:, 2:3]+x[:, 6:7]+x[:, 10:11]+x[:, 14:15]
            qx2, qy1 = -pTx-qx1, -pTy-qy2

            # Concatenate energy and momentum components.
            qc1, qc2 = torch.cat([Eq1, qx1, qy1, qz1], 1), torch.cat([Eq2, qx2, qy2, qz2], 1)

        elif self.learn_mode is None:
            # Similar to 'mc' mode but energies are computed from momenta.
            qx1, qy2, qz1, qz2 = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
            pTx, pTy = x[:, 1:2]+x[:, 5:6]+x[:, 9:10]+x[:, 13:14], x[:, 2:3]+x[:, 6:7]+x[:, 10:11]+x[:, 14:15]
            qx2, qy1 = -pTx-qx1, -pTy-qy2
            Eq1, Eq2 = torch.sqrt(self.m_C**2 + qx1**2 + qy1**2 + qz1**2), torch.sqrt(self.m_C**2 + qx2**2 + qy2**2 + qz2**2)
            qc1, qc2 = torch.cat([Eq1, qx1, qy1, qz1], 1), torch.cat([Eq2, qx2, qy2, qz2], 1)

        # Calculate combined momenta for different particle combinations.
        pB1, pB2 = pb1 + qc1, pb2 + qc2
        pA1, pA2 = pa1 + pB1, pa2 + pB2
        pT = (pA1 + pA2)[:, 1:3]

        # Compute squared masses if flag is set, else compute regular masses.
        if self.mass_squared:
            mC1_sq, mC2_sq, mB1_sq, mB2_sq, mA1_sq, mA2_sq = squared_mass(qc1), squared_mass(qc2), squared_mass(pB1), squared_mass(pB2), squared_mass(pA1), squared_mass(pA2)
        else:
            mC1_sq, mC2_sq, mB1_sq, mB2_sq, mA1_sq, mA2_sq = mass_torch(qc1), mass_torch(qc2), mass_torch(pB1), mass_torch(pB2), mass_torch(pA1), mass_torch(pA2)

        # Update model's mass attributes.
        self.compute_mass()

        # Create tensors with expected masses for particles.
        mCs, mBs, mAs = self.m_C * torch.ones_like(mC1_sq), self.m_B * torch.ones_like(mB1_sq), self.m_A * torch.ones_like(mA1_sq)

        # Define the metric for loss computation.
        metric = self.dist_fn

        # Compute individual losses based on discrepancies between predicted and expected masses and momenta.
        loss_C = metric(mC1_sq, mC2_sq) + metric(mC1_sq, mCs) + metric(mC2_sq, mCs)
        loss_B = metric(mB1_sq, mB2_sq) + metric(mB1_sq, mBs) + metric(mB2_sq, mBs)
        loss_A = metric(mA1_sq, mA2_sq) + metric(mA1_sq, mAs) + metric(mA2_sq, mAs)
        loss_pT = pT[:, 0]**2 + pT[:, 1]**2

        # Aggregate losses based on the learning mode.
        if self.learn_mode == 'pt_mc':
            loss = (loss_A + loss_B + loss_C).mean() + loss_pT.mean()
        elif self.learn_mode == 'pt':
            loss = (loss_A + loss_B).mean() + loss_pT.mean()
        elif self.learn_mode == 'mc':
            loss = (loss_A + loss_B + loss_C).mean()
        else:
            loss = (loss_A + loss_B).mean()

        # Log important metrics for tracking during training.
        self.log('loss', loss, prog_bar=True)
        self.log('m_A', self.m_A, prog_bar=True)
        self.log('m_B', self.m_B, prog_bar=True)

        # Return the aggregate loss.
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
        # Initialize the Adam optimizer with the model's parameters and specified learning rate.
        optimizer = optim.Adam(
            self.parameters(),       # Parameters to optimize
            lr=self.learning_rate,   # Learning rate for the optimizer
        )

        # Return a dictionary with the optimizer and learning rate scheduler configurations.
        return {
            # The optimizer to be used for model parameter updates.
            "optimizer": optimizer,

            # Learning rate scheduler and its associated configuration.
            "lr_scheduler": {
                # Use Polynomial learning rate scheduler, which adjusts the learning rate
                # based on a polynomial of the provided degree ('power').
                "scheduler": PolynomialLR(
                    optimizer,                  # Associated optimizer
                    total_iters=self.epochs,    # Total number of iterations (epochs)
                    power=2.0                   # Degree of the polynomial
                ),
                
                # Scheduler update interval - here it's set to update every epoch.
                "interval": "epoch",
                
                # The metric to monitor for learning rate adjustment. In this case, it's the validation loss.
                "monitor": "val_loss",
                
                # If set to True, scheduler raises an error if the metric (val_loss) is not found. 
                # This ensures that the monitored metric is correctly logged and available.
                "strict": True,
            }
        }

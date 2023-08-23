import torch
from torch.utils.data import DataLoader, random_split
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np

from deelema import DeeLeMa, ToyData

import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42)

# User defined scale parameter (prevents overflow)
SCALE_PARAM = 1000
AVAIL_GPUS  = min(1, torch.cuda.device_count())
BATCH_SIZE  = 2048 if AVAIL_GPUS else 64

def main():
    # Load data
    DATA_PATH = "data/"
    PROCESS   = "toy"
    np_data   = np.load(DATA_PATH + PROCESS + ".npz")

    pa1 = np_data['b1']
    pa2 = np_data['b2']
    pb1 = np_data['l2']
    pb2 = np_data['l1']
    qc1 = np_data['nu1']
    qc2 = np_data['nu2']

    X = np.concatenate((pa1, pa2, pb1, pb2, qc1, qc2), axis=1)
    torch_momenta = list(
            map(lambda x: torch.tensor(x/SCALE_PARAM, dtype=torch.float32), np.array_split(X, 6, axis=1))
    )

    # Create dataset
    ds = ToyData(*torch_momenta)
    N  = len(ds)

    # Split dataset
    N_train = int(0.7*N)
    N_val   = int(0.2*N)
    N_test  = N - N_train - N_val
    train_ds, val_ds, test_ds = random_split(ds, [N_train, N_val, N_test])

    # Create dataloaders
    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Hyperparameters
    m_C = 700 / SCALE_PARAM
    hparams = {
        "hidden_nodes": 256,
        "hidden_layers": 5,
        "learning_rate": 1e-2,
        "batch_size": BATCH_SIZE,
        "m_C_init": m_C,
        "m_B_add": 0.3,
        "m_A_add": 0.3,
        "epochs": 100,
        "dist_fn": torch.nn.HuberLoss(),
        "learn_mode": None,
        "mass_squared": False,
    }

    # Create model
    model = DeeLeMa(hparams=hparams)

    # Create logger
    logger = TensorBoardLogger(save_dir="logs/")

    # Create trainer
    trainer = Trainer(
        devices=AVAIL_GPUS,
        accelerator="auto",
        logger=logger,
        max_epochs=hparams["epochs"],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ],
    )

    # Train model
    trainer.fit(model, dl_train, dl_val)

    # Save model
    trainer.save_checkpoint("DeeLeMa_Toy.ckpt")

    # Add custom test & analyze code
    # ...

if __name__ == "__main__":
    main()

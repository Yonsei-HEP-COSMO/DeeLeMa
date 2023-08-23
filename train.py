# Necessary imports for training
import torch
from torch.utils.data import DataLoader, random_split
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np

# Custom modules from the DeeLeMa package
from deelema import DeeLeMa, ToyData

# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set global PyTorch settings
torch.set_float32_matmul_precision('medium')
pl.seed_everything(42)

# Global constants for scaling and hardware utilization
SCALE_PARAM = 1000
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 2048 if AVAIL_GPUS else 64

def main():
    # Load data
    DATA_PATH = "data/"
    PROCESS = "toy"
    
    # Load .npz file and split into individual numpy arrays
    np_data = np.load(DATA_PATH + PROCESS + ".npz")

    pa1 = np_data['b1']
    pa2 = np_data['b2']
    pb1 = np_data['l2']
    pb2 = np_data['l1']
    qc1 = np_data['nu1']
    qc2 = np_data['nu2']

    # Concatenate data and split into torch tensors, scaled by SCALE_PARAM
    X = np.concatenate((pa1, pa2, pb1, pb2, qc1, qc2), axis=1)
    torch_momenta = list(
        map(
            lambda x: torch.tensor(x/SCALE_PARAM, dtype=torch.float32),
            np.array_split(X, 6, axis=1)
        )
    )

    # Create a dataset instance using the ToyData class
    ds = ToyData(*torch_momenta)
    N = len(ds)

    # Split dataset into training, validation, and testing parts
    N_train = int(0.7*N)
    N_val = int(0.2*N)
    N_test = N - N_train - N_val
    train_ds, val_ds, test_ds = random_split(ds, [N_train, N_val, N_test])

    # Create dataloaders for each dataset split
    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Hyperparameter setup for the model
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

    # Instantiate the DeeLeMa model
    model = DeeLeMa(hparams=hparams)

    # Setup TensorBoard for logging training progress
    logger = TensorBoardLogger(save_dir="logs/")

    # Configure the Trainer with relevant settings, callbacks, and logging utilities
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

    # Train the model
    trainer.fit(model, dl_train, dl_val)

    # Save the trained model's checkpoint
    trainer.save_checkpoint("DeeLeMa_Toy.ckpt")

    # Space reserved for custom testing & analysis code
    # ...

if __name__ == "__main__":
    # Run the main function if the script is executed as a standalone file
    main()
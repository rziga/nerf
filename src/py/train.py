import argparse
from pathlib import Path

import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data.datamodule import BlenderDataModule
from model.lightningmodule import MobileR2LLighningModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MobileR2L training script.")
    parser.add_argument("--root", type=Path, required=True, help="Path to pseudo dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint dir.")
    parser.add_argument("--seed", type=int, required=False, default=1337, help="Seed for the experiment.")
    parser.add_argument("--batch_size", type=int, required=False, default=4, help="Batch size for the experiment.")
    parser.add_argument("--name", required=False, default="", help="Name for the experiment.")
    args = parser.parse_args()

    L.seed_everything(args.seed)
    datamodule = BlenderDataModule(args.root, args.batch_size, 16, rand=True)
    model = MobileR2LLighningModule(5e-4, 16*3, 10, 256, 16, 3)
    trainer = L.Trainer(
        logger = WandbLogger("MobileR2L", project="NeLF"),
        callbacks=[
            ModelCheckpoint(
                dirpath=args.checkpoint,
                monitor="val_psnr", mode="max",
                every_n_train_steps=100
            )
        ],
        max_steps=10_000
    )

    trainer.fit(
        model,
        datamodule,
    )
    trainer.test(
        model,
        datamodule
    )

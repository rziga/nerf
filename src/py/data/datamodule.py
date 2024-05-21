from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, random_split

from .dataset import BlenderPseudoDataset


class BlenderDataModule(L.LightningDataModule):
    
    def __init__(self, root, batch_size, num_ray_samples, rand):
        super().__init__()
        self.save_hyperparameters("batch_size", "num_ray_samples", "rand")
        self.root = Path(root)

    def setup(self, stage):
        if stage == "fit":
            trainval = BlenderPseudoDataset(
                self.root, 
                self.hparams["num_ray_samples"],
                self.hparams["rand"]
            )
            self.train, self.val = random_split(trainval, [0.8, 0.2])
        elif stage == "test":
            self.test = BlenderPseudoDataset(
                self.root, self.hparams["num_ray_samples"], rand=False
            )
        elif stage == "predict":
            pass
    
    def train_dataloader(self):
        return DataLoader(
            self.train, self.hparams["batch_size"], shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test)

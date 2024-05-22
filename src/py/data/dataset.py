import json
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np

from .utils import get_rays, get_c2w


class BlenderPseudoDataset(Dataset):

    def __init__(self, root, num_samples, rand) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.rand = rand

        root = Path(root)
        self.dataset_info = self._load_info(root/"hwf"/"dataset_info.json")
        print("searching for images...")
        self.img_paths = sorted([f for f in root.glob("vis/*") if "_" not in f.stem], key=lambda f: int(f.stem))
        print("loading poses...")
        self.poses = [get_c2w(self._load(f)) for f in sorted(root.glob("poses/*"), key=lambda f: int(f.stem.split("_")[1]))]
        print("done")

        if len(self.img_paths) != len(self.poses):
            warnings.warn(f"Number of images does not match number of poses in the dataset at {root}.")

    def __getitem__(self, index):
        img = read_image(str(self.img_paths[index])) / 255
        pose = self.poses[index]
        rays = get_rays(
            self.dataset_info["downscaled_height"],
            self.dataset_info["downscaled_width"],
            self.dataset_info["downscaled_focal"],
            pose,
            0, 1,
            self.num_samples,
            self.rand
        )
        return rays, img
    
    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _load(fpath):
        return torch.tensor(np.load(fpath))
    
    @staticmethod
    def _load_info(fpath):
        with open(fpath, "r") as f:
            info = json.load(f)
        return info
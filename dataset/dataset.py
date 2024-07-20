from typing import Sequence, Dict, Union
import math
import time

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
from torchvision import transforms as T

class DiffJSCCDataset(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
    ) -> "DiffJSCCDataset":
        super(DiffJSCCDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip

        transforms = []
        if self.crop_type == 'random':
            transforms.append(T.RandomResizedCrop(size=(out_size, out_size)))
        elif self.crop_type == 'center':
            transforms.append(T.CenterCrop(size=(out_size, out_size)))

        if self.use_hflip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        transforms.append(T.ToTensor())
        self.transforms = T.Compose(transforms)        

 
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        
        target = self.transforms(pil_img)
        
        return dict(jpg=target, txt="")

    def __len__(self) -> int:
        return len(self.paths)

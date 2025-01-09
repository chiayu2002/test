import glob
import numpy as np
from PIL import Image
import os
import torch

from torchvision.datasets.vision import VisionDataset



class ImageDataset(VisionDataset):
    """
    Load images from multiple data directories.
    Folder structure: data_dir/filename.png
    """

    def __init__(self, data_dirs, transforms=None, label_file=None):
        # Use multiple root folders
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]

        # initialize base class
        VisionDataset.__init__(self, root=data_dirs, transform=transforms)

        self.filenames = []
        root = []
        self.labels = {}

        category_map = {
            "0.5_": 0,  
            "1_": 1,  
            "2_": 2   
        }
        for dir_idx, ddir in enumerate(self.root):
            filenames = self._get_files(ddir)
            self.filenames.extend(filenames)

            for filename in filenames:
                for category_prefix, category_idx in category_map.items():
                    if filename.startswith(f"{ddir}/{category_prefix}"):
                        file_idx = int(filename.split('/')[-1].replace(category_prefix, "").replace('.jpg', '').lstrip('0')) - 1
                        self.labels[filename] = [dir_idx, category_idx, file_idx]
                        break 
            root.append(ddir)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_files(root_dir):
        return glob.glob(f'{root_dir}/*.png') + glob.glob(f'{root_dir}/*.jpg' )+ glob.glob(f'{root_dir}/*.PNG')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels.get((filename), 0)
        label = torch.tensor(label, dtype=torch.float32)
        device = img.device
        label = label.to(device)
        return img, label

class Carla(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(Carla, self).__init__(*args, **kwargs)
    
class RS307_0_i2(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(RS307_0_i2, self).__init__(*args, **kwargs)
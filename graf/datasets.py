import glob
import numpy as np
from PIL import Image
import os
import torch
#import pandas as pd

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
                        #print(f"1label:{file_idx}")
                        self.labels[filename] = [dir_idx, category_idx, file_idx]
                        #print(f"label:{self.labels}")
                        break 
            root.append(ddir)
        
        # label_file = '/Data/home/vicky/graf-main/data/damageindex.csv'
        # if os.path.exists(label_file):
        #     df = n.read_csv(label_file)
        #     for _, row in df.iterrows():
        #         filename = os.path.join(ddir, row['filename'])
        #         label = (row['DI'], row['AR'], row['VR'], row['HR'])
        #         self.labels[filename] = label
        # if label_file:
        #     self.labels = self.load_labels(label_file)   

    # def load_labels(self, label_file):
    #     import pandas as pd
    #     df = pd.read_csv(label_file)
    #     return {row['filename']: row['DI'] for _, row in df.iterrows()} 

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
        #print(f"img shape: {img.shape}")
       #print(f"label shape: {label.shape}")
        device = img.device
        label = label.to(device)
        # label_expanded = label.unsqueeze(1).unsqueeze(2).expand(-1, img.shape[1], img.shape[2])
        # combined = torch.cat([img, label_expanded], dim=0)
        return img, label

class Carla(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(Carla, self).__init__(*args, **kwargs)


class CelebA(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CelebA, self).__init__(*args, **kwargs)


class CUB(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CUB, self).__init__(*args, **kwargs)
        

class Cats(ImageDataset):
    def __init__(self, *args, **kwargs):
      super(Cats, self).__init__(*args, **kwargs)
    
    @staticmethod
    def _get_files(root_dir):
      return glob.glob(f'{root_dir}/CAT_*/*.jpg')
    
class RS307_0_i2(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(RS307_0_i2, self).__init__(*args, **kwargs)

class images_2(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(images_2, self).__init__(*args, **kwargs)

class CelebAHQ(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CelebAHQ, self).__init__(*args, **kwargs)
    
    def _get_files(self, root):
        return glob.glob(f'{root}/*.npy')
    
    def __getitem__(self, idx):
        img = np.load(self.filenames[idx]).squeeze(0).transpose(1,2,0)
        if img.dtype == np.uint8:
            pass
        elif img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        else:
            raise NotImplementedError
        img = Image.fromarray(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img

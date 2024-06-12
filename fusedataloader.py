from __future__ import print_function,division
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path

class Imagedata(Dataset):
    def __init__(self, image_path, spec_path, transform=None):
        super(Imagedata, self).__init__()
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(image_path)
        self.samples = self.make_dataset(image_path, spec_path, self.class_to_idx)
        self.targets = [s[2] for s in self.samples]

    def _find_classes(self, dir):
        classes = []
        for d in Path(dir).rglob("*"):
            if d.is_dir():
                parts = d.parts # 
                item = f"{parts[-1]}"
                classes.append(item)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def _get_target(self, file_path):
        parts = file_path.parts
        target_class = f"{parts[-2]}"
        return target_class

    def make_dataset(self, img_dir, spec_dir, class_to_idx):
        images = []
        for d in os.listdir(img_dir):
            for image in os.listdir(img_dir+'/'+d):
                img= img_dir+'/'+d+'/'+image
                spec= spec_dir+'/'+d+'/'+image
                spec = spec.replace('_Image','_Movement')
                spec = spec.replace('.jpg','.png')
                item = (img, spec, class_to_idx[d])
                images.append(item)
        return images

    def get_class_dict(self):
        """Returns a dictionary of classes mapped to indicies."""
        return self.class_to_idx
    def __getitem__(self, index):
        img, spec, target = self.samples[index]
        img = Image.open(img)
        spec = Image.open(spec)
        if spec.mode == 'RGBA':
            spec = spec.convert("RGB")
        if img.mode == 'L':
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            spec = self.transform(spec)
        return img, spec, target
 
    def __len__(self):
        return len(self.samples)



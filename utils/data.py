import os
import glob
import numpy as np
from PIL import Image
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.4,\
            iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5)),
        iaa.Sometimes(0.4,\
            iaa.GaussianBlur(sigma=(0, 0.2))),
    ])
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    return img

class ImageDataset(Dataset):
    def __init__(self, paths, is_aug=True):
        super(ImageDataset, self).__init__()

        # Length
        self.length = len(paths)
        # Image path
        self.paths = paths
        # Augment
        self.is_aug = is_aug
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgAugTransform(),
            lambda x: Image.fromarray(x),
        ])
        # Preprocess
        self.output = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Image
        img = Image.open(self.paths[idx])
        # Augment
        if self.is_aug:
            img = self.transform(img)
        # Preprocess
        img = self.output(img)

        return img

def get_celeba_loaders(batch_train, batch_test):
    images = glob.glob(os.path.join(".", "data", "celeba", "*.jpg"))
    train, test = train_test_split(images, test_size=0.1, random_state=42, shuffle=True)
    
    datasets = {
        "train": ImageDataset(train, True),
        "test": ImageDataset(test, False)
    }
    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_train, shuffle=True),
        "test": DataLoader(datasets["test"], batch_size=batch_test, shuffle=False)
    }

    return dataloaders
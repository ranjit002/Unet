import os

import albumentations as A
import torch
from numpy import float32
from tifffile import imread
from torch.utils.data import DataLoader, Dataset, random_split


class TiffDataset(Dataset):
    def __init__(self, data_path, mask_path, transform=None):
        """
        Class that parses .tiff images and transforms them into training data.
        The images are only loaded and transformed as they are needed, to lower memory load.
        NOTE: data and target imgs must have same file names in the directories.

        Args:
            data_path (str): path of file with training images (data)
            mask_path (str): path of file with masks
            transform (torch/albumentations transform): img transform. Defaults to None.
        """
        self.data_path = data_path
        self.mask_path = mask_path

        self.image_filenames = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of data of image in directory

        Output:
            data_image (tensor): transformed data img
            mask_image (tensor): transformed mask img
        """
        data_path = os.path.join(self.data_path, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_path, self.image_filenames[idx])

        data_image = imread(data_path)
        mask_image = imread(mask_path)

        data_image = data_image.astype(float32) / 65535.0  # For 16-bit .tif images
        mask_image = (mask_image > 0) * 1

        if self.transform:
            transformed = self.transform(image=data_image, mask=mask_image)
            data_image, mask_image = transformed["image"], transformed["mask"]

        return data_image, mask_image


data_transform = A.Compose(
    transforms=[
        A.Rotate(  # Rotate image by random angle whilst retaining 256x256 shape
            limit=180, p=1.0
        ),
    ]
)

# Paths to the data and mask folders
DATA_DIR = "./dataset/train_imgs"
MASK_DIR = "./dataset/train_masks"


dataset = TiffDataset(DATA_DIR, MASK_DIR, data_transform)

train_dataset, valid_dataset = random_split(
    dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(25)
)

# Create dataloaders to iterate over minibatches
BATCH_SIZE = 20

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

val_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

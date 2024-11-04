import os

from albumentations import Compose, Rotate
from numpy import float32, mean, nonzero
from tifffile import imread  # Import tifffile for reading .tif images
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split


class TiffDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir

        self.image_filenames = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])

        data_image = imread(data_path)
        mask_image = imread(mask_path)

        data_image[data_image < 10000] = mean(data_image[nonzero(data_image)])

        data_image = data_image.astype(float32) / 65535.0  # For 16-bit .tif images
        mask_image = (mask_image > 0) * 1

        if self.transform:
            transformed = self.transform(image=data_image, mask=mask_image)
            data_image, mask_image = transformed["image"], transformed["mask"]

        return data_image, mask_image


img_transform = Compose(
    transforms=[
        Rotate(limit=180, p=1.0),
        # Albumentations.RandomRotate90(p=1.0),
    ]
)

# Paths to the data and mask folders
DATA_DIR = "./seep_detection/train_images_256"
MASK_DIR = "./seep_detection/train_masks_256"

# Create the dataset
dataset = TiffDataset(DATA_DIR, MASK_DIR, img_transform)

train_dataset, valid_dataset = random_split(
    dataset,
    [0.9, 0.1],
    generator=Generator().manual_seed(25),
)


BATCH_SIZE = 20

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)

val_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)

import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import cv2
import yaml
import os


class I2IDataset(data.Dataset):
    def __init__(self, cfg, train=True):
        """
        Initializes the dataset with training/testing data and augmentations.

        Args:
            cfg (dict): Configuration dictionary loaded from YAML.
            train (bool): Flag to indicate if the dataset is for training or testing.
        """
        self.is_train = train
        self.cfg = cfg
        self.A_imgs, self.B_imgs = self.load_data()

        # Augmentations
        self.gan_aug = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=cfg['Translation']['augmentation']['shift_limit'],
                scale_limit=cfg['Translation']['augmentation']['scale_limit'],
                rotate_limit=cfg['Translation']['augmentation']['rotate_limit'],
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.VerticalFlip(p=cfg['Translation']['augmentation']['vertical_flip_prob']),
            A.HorizontalFlip(p=cfg['Translation']['augmentation']['horizontal_flip_prob']),
            A.Normalize(
                mean=(cfg['Translation']['augmentation']['mean'],),
                std=(cfg['Translation']['augmentation']['std'],),
                max_pixel_value=cfg['Translation']['augmentation']['max_pixel_value']
            ),
            ToTensorV2()
        ])

    def load_data(self):
        """
        Loads the training or target datasets from the paths specified in the configuration.

        Returns:
            tuple: A_imgs and B_imgs as NumPy arrays.
        """

        
        data_type = 'train' if self.is_train else 'val'
 
        A_imgs = np.load(os.path.join(self.cfg['experiment_name'], "Translation", "translation_data", data_type, 'source_imgs.npy'))
        B_imgs = np.load(os.path.join(self.cfg['experiment_name'], "Translation", "translation_data", data_type, 'target_imgs.npy'))
        
        
        return A_imgs, B_imgs

    def __getitem__(self, index):
        """
        Fetches a sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            dict: Contains augmented 'A_img' and 'B_img' tensors.
        """
        A_img = self.A_imgs[index]
        B_index = random.randint(0, self.B_imgs.shape[0] - 1)
        B_img = self.B_imgs[B_index]

        # Apply augmentations
        A_img = self.gan_aug(image=A_img)["image"]
        B_img = self.gan_aug(image=B_img)["image"]
        return {'A_img': A_img, 'B_img': B_img}

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.A_imgs.shape[0]


def load_config(config_path):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config.yaml")

    # Initialize dataset
    dataset = I2IDataset(cfg, train=True)

    # Example: Fetch a sample
    sample = dataset[0]
    print(f"Sample A_img shape: {sample['A_img'].shape}, B_img shape: {sample['B_img'].shape}")

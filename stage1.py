import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse


from utils import I2IDataset, create_dirs, load_config
from Translation.i2i_solver import i2iSolver





def check_manual_seed(seed):
    """
    Sets a manual seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #print(f"Using manual seed: {seed}")


def setup_device(cfg):
    """
    Sets up the GPU or CPU device based on configuration.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        torch.device: The device to use for computation.
    """
    device_id = cfg['gpu']['device']
    if torch.cuda.is_available() and device_id >= 0:
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        return torch.device(f"cuda:{device_id}")

    else:
        print("GPU not available or device ID is invalid. Using CPU.")
        return torch.device("cpu")


# Main Function
def main():
    """
    Main function to initialize and train the I2I GAN model.
    """
    parser = argparse.ArgumentParser(description="Train an I2I GAN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file.")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Set up environment
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    check_manual_seed(cfg["Translation"]['model']['seed'])
    create_dirs(cfg["Translation"]['model']['name'])

    # Set up device
    device = setup_device(cfg)

    # Load data
    train_loader = DataLoader(
        dataset=I2IDataset(cfg, train=True),
        batch_size=cfg["Translation"]['data']['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["Translation"]['data']['num_workers'],
        pin_memory=True
    )
    validation_loader = DataLoader(
        dataset=I2IDataset(cfg, train=False),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=cfg["Translation"]['data']['num_workers'],
        pin_memory=True
    )

    # Initialize solver
    trainer = i2iSolver(cfg["Translation"]['model'])
    trainer.to(device)
    iteration = 0

    # Training loop
    for epoch in range(cfg["Translation"]['model']['epochs']):
        for train_data in train_loader:
            # Move data to device
            train_data = {k: v.to(device).detach() for k, v in train_data.items()}

            # GAN forward and updates
            trainer.gan_forward(train_data['A_img'], train_data['B_img'])
            trainer.dis_update()
            trainer.gen_update()

            # Verbose output
            text = trainer.verbose()
            if iteration % cfg["Translation"]['data']['visual_interval'] == 0:
                trainer.gan_visual(epoch)

            sys.stdout.write(f'\r Epoch {epoch}, Iter {iteration}, {text}')
            iteration += 1

        # Save model at the end of each epoch
        trainer.save(epoch)


if __name__ == '__main__':
    main()

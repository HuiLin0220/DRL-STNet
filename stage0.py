import os
import argparse
import random
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import load_config

def adaptive_normalize(arr):
    """
    Normalizes the input array by clipping values to the 99.99th percentile
    of non-zero elements and scaling to the range [0, 1].
    """
    max_p = 1 - 0.0001 * arr.shape[-1]
    arr = arr.astype(np.float32)
    PixelArr = arr[arr > 0]
    if len(PixelArr) > 0:
        PixelArr.sort()
        max_v = PixelArr[int((len(PixelArr) - 1) * max_p + 0.5)]
        arr = np.clip(arr, 0, max_v) / max_v
    return arr


def random_select_and_save(input_path, output_file, num_samples=50):
    """
    Randomly selects files from the given directory and saves the selected file names to a text file.

    Args:
        input_path (str): Path to the directory containing files.
        output_file (str): Path to the output text file.
        num_samples (int): Number of files to select.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The directory {input_path} does not exist.")

    all_files = os.listdir(input_path)
    if len(all_files) == 0:
        print(f"No files found in directory {input_path}. Skipping.")
        return []

    randomly_selected_files = random.sample(all_files, min(len(all_files), num_samples))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for file_name in tqdm(randomly_selected_files, desc=f"Saving to {output_file}"):
            f.write(file_name + "\n")

    print(f"Randomly selected file names saved to {output_file}.")
    return randomly_selected_files

'''
def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process and normalize NIfTI images.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing 'source' and 'target' folders."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the processed .npy files and selected files."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Target size for image resizing (default: 512)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of files to randomly select from each folder (default: 50)."
    )
    return parser.parse_args()
'''
def parse_args():
    """
    Parses command-line arguments.
    """   
    parser = argparse.ArgumentParser(description="Process and normalize NIfTI images.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file.")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    return cfg


def process_files(file_list, folder_path, img_size):
    """
    Processes NIfTI files into numpy arrays.

    Args:
        file_list (list): List of file names to process.
        folder_path (str): Path to the folder containing the files.
        img_size (int): Target size for image resizing.

    Returns:
        list: Processed numpy arrays.
    """
    processed_images = []
    for file_name in tqdm(file_list, desc=f"Processing files in {folder_path}"):
        file_path = os.path.join(folder_path, file_name)
        try:
            if not os.path.isfile(file_path) or not file_name.endswith((".nii", ".nii.gz")):
                continue  # Skip non-NIfTI files

            img = nib.load(file_path)
            image_data = img.get_fdata().transpose((1, 0, 2))[::-1]
            origin_shape = image_data.shape
            image_data = torch.from_numpy(image_data.copy()).unsqueeze(0).unsqueeze(0)
            image_data = F.interpolate(
                image_data,
                [img_size, img_size, origin_shape[-1]],
                mode="trilinear"
            ).numpy()[0, 0]
            image_data = adaptive_normalize(image_data)

            processed_images.append(image_data)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue
    return processed_images


def main():
    """
    Main function to process NIfTI files.
    """
    cfg = parse_args()


    raw_data_path = cfg["Translation"]["raw_data_path"]
    
    output_path =  os.path.join(cfg["experiment_name"], "Translation")
    
    num_samples = cfg["Translation"]["num_samples"]
    img_size = cfg["Translation"]["img_size"]
    
    
    # Define source and target paths
    source_path = os.path.join(raw_data_path, "source")
    target_path = os.path.join(raw_data_path, "target")

    # Define output file paths for selected files
    source_output_file = os.path.join(output_path, "source_selected_files.txt")
    target_output_file = os.path.join(output_path, "target_selected_files.txt")

    # Randomly select and save files from source and target paths
    source_file_list = random_select_and_save(source_path, source_output_file, num_samples)
    target_file_list = random_select_and_save(target_path, target_output_file, num_samples)

    # Process and save source files as A_imgs
    A_imgs = process_files(source_file_list, source_path, img_size)
    if A_imgs:
        A_imgs = np.concatenate(A_imgs, axis=-1).transpose((2, 0, 1))
        np.save(os.path.join(output_path, "source_imgs.npy"), A_imgs)
        print(f"A_imgs saved to {os.path.join(output_path, 'source_imgs.npy')}")

    # Process and save target files as B_imgs
    B_imgs = process_files(target_file_list, target_path, img_size)
    if B_imgs:
        B_imgs = np.concatenate(B_imgs, axis=-1).transpose((2, 0, 1))
        np.save(os.path.join(output_path, "target_imgs.npy"), B_imgs)
        print(f"B_imgs saved to {os.path.join(output_path, 'target_imgs.npy')}")


if __name__ == "__main__":
    main()

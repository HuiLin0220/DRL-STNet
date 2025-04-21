import os
import yaml
import torch
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F
from Translation.i2i_solver import i2iSolver
from utils import load_config, adaptive_normalize


# ------------------------- Image Preprocessing ------------------------- #
def process(nii_path):
    """Load, normalize, and resize a NIfTI image."""
    imgs = nib.load(nii_path).get_fdata().transpose((1, 0, 2))[::-1]
    origin_shape = imgs.shape
    image_data = torch.from_numpy(imgs.copy()).unsqueeze(0).unsqueeze(0)
    image_data = F.interpolate(image_data, [512, 512, origin_shape[-1]], mode="trilinear").numpy()[0, 0]
    return adaptive_normalize(image_data), origin_shape


# ------------------------- Main Pipeline ------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    # ------------------------- Load Configuration ------------------------- #
    cfg = load_config(args.config)
    pred_cfg = cfg["Translation"]["prediction"]
    prep_cfg = cfg["Translation"]["preparation"]

    # Model and clustering configuration
    ckpt_epoch = cfg["Translation"]["model"].get("epoch", 40)
    ckpt_filename = f"enc_{ckpt_epoch:04d}.pt"
    ckpt_path = os.path.join(cfg['experiment_name'], "Translation", cfg["Translation"]['model']['name'], "i2i_checkpoints", ckpt_filename)

    k_means_clusters = pred_cfg["k_means_clusters"]

    # Construct data paths
    raw_data_path = prep_cfg["raw_data_path"]
    source_dir = os.path.join(raw_data_path, "source")
    target_dir = os.path.join(raw_data_path, "target")

    # Output directory
    save_dir = pred_cfg["save_nii_dirpath"]
    os.makedirs(save_dir, exist_ok=True)

    # Sample sizes for random selection
    num_source = pred_cfg["num_samples"]["source"]
    num_target = pred_cfg["num_samples"]["target"]

    # Randomly select subset of files
    all_source_files = [f for f in os.listdir(source_dir) if f.endswith(".nii.gz")]
    all_target_files = [f for f in os.listdir(target_dir) if f.endswith(".nii.gz")]
    selected_source_files = random.sample(all_source_files, num_source)
    selected_target_files = random.sample(all_target_files, num_target)

    # Save selected filenames for reproducibility
    with open(os.path.join(save_dir, "selected_source_files.txt"), 'w') as f:
        f.writelines(f"{name}\n" for name in selected_source_files)

    with open(os.path.join(save_dir, "selected_target_files.txt"), 'w') as f:
        f.writelines(f"{name}\n" for name in selected_target_files)

    # ------------------------- Load Pretrained Model ------------------------- #
    trainer = i2iSolver(None)
    state_dict = torch.load(ckpt_path)
    trainer.enc_c.load_state_dict(state_dict['enc_c'])
    trainer.enc_s_a.load_state_dict(state_dict['enc_s_a'])
    trainer.enc_s_b.load_state_dict(state_dict['enc_s_b'])
    trainer.dec.load_state_dict(state_dict['dec'])
    trainer.cuda()

    # ------------------------- Extract Style Vectors ------------------------- #
    print('========= Extracting styles from randomly selected target images =========')
    styles = []
    for fname in tqdm(selected_target_files):
        nii_path = os.path.join(target_dir, fname)
        imgs, _ = process(nii_path)
        for i in range(int(imgs.shape[-1] / 6), int(imgs.shape[-1] / 6 * 5)):
            img = imgs[:, :, i]
            with torch.no_grad():
                input_tensor = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
                style_vec = trainer.enc_s_b(input_tensor).cpu().numpy()[0]
                styles.append(style_vec)

    # ------------------------- Perform K-Means Clustering ------------------------- #
    print('========= Performing K-Means clustering =========')
    k_mean_results = KMeans(n_clusters=k_means_clusters, random_state=9).fit_predict(styles)

    # ------------------------- Apply Style Transfer to Source Images ------------------------- #
    print('========= Transferring styles to randomly selected source images =========')

    for fname in tqdm(selected_source_files):
        nii_path = os.path.join(source_dir, fname)
        imgs, origin_shape = process(nii_path)
        nimgs = np.zeros_like(imgs, dtype=np.float32)

        idx = random.choice(np.argwhere(k_mean_results == 0).flatten().tolist())
        s = torch.from_numpy(styles[idx]).unsqueeze(0).cuda().float()

        for i in range(imgs.shape[-1]):
            img = imgs[:, :, i]
            input_tensor = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
            with torch.no_grad():
                output_tensor = trainer.inference(input_tensor, s)
            nimgs[:, :, i] = (((output_tensor + 1) / 2).cpu().numpy())[0, 0]

        # Resize back to original shape and save
        image_data = torch.from_numpy(nimgs.copy()).unsqueeze(0).unsqueeze(0)
        image_data = F.interpolate(image_data, [origin_shape[0], origin_shape[1], origin_shape[-1]], mode="trilinear").numpy()[0, 0]
        image_data = np.rot90(image_data, k=-1, axes=(0, 1))

        affine = nib.load(nii_path).affine
        out_nii = nib.Nifti1Image(image_data, affine)
        out_name = fname.replace(fname.split('_')[-1], '0000.nii.gz')
        nib.save(out_nii, os.path.join(save_dir, out_name))


if __name__ == "__main__":
    main()

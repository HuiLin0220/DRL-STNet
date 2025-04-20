# DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning
[![paper](https://img.shields.io/badge/arXiv-2409.18340-blue)](https://arxiv.org/abs/2409.18340)
 [![cite](https://img.shields.io/badge/cite-BibTex-yellow)](https://scholar.googleusercontent.com/scholar.bib?q=info:_WTBYxMIx-IJ:scholar.google.com/&output=citation&scisdr=ClEVFUEBEJ750VrYKyA:AFWwaeYAAAAAZ-beMyBvhn9GPawRgNF6OIXYJOY&scisig=AFWwaeYAAAAAZ-beM5qZ-hFUtE2RQAeeiZ-cNo0&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1) 

# Introduction
This algorithm is for Task 3 in [FLARE Challenge](https://www.codabench.org/competitions/2296/), which was held at MICCAI 2024. We are ranked ${\textsf{\color{red}5th}}$ regarding the accuracy!


If you find our work is useful in your research, please consider citing:

(1) [DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning](https://arxiv.org/abs/2409.18340)
```bash
@article{lin2024drl,
  title={DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning},
  author={Lin, Hui and Schiffers, Florian and L{\'o}pez-Tapia, Santiago and Tavakoli, Neda and Kim, Daniel and Katsaggelos, Aggelos K},
  journal={arXiv preprint arXiv:2409.18340},
  year={2024}
}
```
## Dataset

[To obtain the FLARE Challenge dataset](https://www.codabench.org/competitions/2296/)
## Shared weights



## Folder Structures
- The input for stage 0:
	```
 	FLARE/  
          ├── source
          │   ├── FLARE22_Tr_0001_0000.nii.gz
          │   ├── FLARE22_Tr_0002_0000.nii.gz
          │   ├── ...
          ├── source_labels
          │   ├── FLARE22_Tr_0001.nii.gz
          │   ├── FLARE22_Tr_0002.nii.gz
          │   ├── ...
          ├── target
          │   ├── amos_XXXX_0000.nii.gz
          │   ├── amos_XXXX_0000.nii.gz
          │   ├── ...
	```
- After stage 0 and stage 1
	```
	FLARE/Translation/  
          ├── i2i_exp
          │   ├── i2i_checkpoints
          │   ├── i2i_train_visual
          ├── train
          │   ├── source_imgs.npy
          │   │── target_imgs.npy
          │   │── source_selected_files.txt
          │   │── target_selected_files.txt
          ├── val
          │   ├── source_imgs.npy
          │   │── target_imgs.npy
          │   │── source_selected_files.txt
          │   │── target_selected_files.txt
	```
		
## Instructions

stage 0 preparing data for training i2i model:
```       
python "/home/hln0895/DRL-STNet/stage0.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"
```
stage 1: train i2i model
```	
python "/home/hln0895/DRL-STNet/stage1.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"
```
stage 2: generate fake target scans (MRI)


## Contact Us
Feel free to contact me at huilin2023@u.northwestern.edu

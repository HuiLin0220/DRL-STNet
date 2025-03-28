# DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning
[![paper](https://img.shields.io/badge/arXiv-2409.18340-blue)](https://arxiv.org/abs/2409.18340)
 [![cite](https://img.shields.io/badge/cite-BibTex-yellow)](https://scholar.googleusercontent.com/scholar.bib?q=info:_WTBYxMIx-IJ:scholar.google.com/&output=citation&scisdr=ClEVFUEBEJ750VrYKyA:AFWwaeYAAAAAZ-beMyBvhn9GPawRgNF6OIXYJOY&scisig=AFWwaeYAAAAAZ-beM5qZ-hFUtE2RQAeeiZ-cNo0&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1) 


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

[To obtain the 2018 Atria Segmentation challenge dataset](https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/)

## Prepare data
- The training data folder structure is like this:

         Raw_data/Dataset_train_val/  
          ├── imagesTr
          │   ├── sten_0000_0000.png
          │   ├── sten_0000_0001.png
          │   ├── ...
          │   ├── sten_0001_0000.png      
          │   ├── sten_0001_0001.png      
          │   ├── ... 
          │   ├── sten_0002_0000.png
          │   ├── sten_0002_0001.png
          │   ├── ...
          ├── labelsTr
          │   ├── sten_0000.png
          │   ├── sten_0001.png
          │   ├── sten_0002.png
          │   ├── ...
          ├── dataset.json
## Instructions

stage 0 preparing data for training i2i model:
       
        python "/home/hln0895/DRL-STNet/stage0.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"

stage 1: train i2i model
	
	python "/home/hln0895/DRL-STNet/stage1.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"

stage 2: generate fake target scans (MRI)
## Contact Us
Feel free to contact me at huilin2023@u.northwestern.edu

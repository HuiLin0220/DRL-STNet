# DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning
[![paper](https://img.shields.io/badge/arXiv-2311.12437-blue)](https://doi.org/10.1016/j.heliyon.2024.e28539)
 [![cite](https://img.shields.io/badge/cite-BibTex-yellow)](https://scholar.googleusercontent.com/scholar.bib?q=info:MqXrnsiRsFcJ:scholar.google.com/&output=citation&scisdr=ClEVFUEEEL3snBU4tsM:AFWwaeYAAAAAZqU-rsMC6F0E_6wELePdP-1rjCA&scisig=AFWwaeYAAAAAZqU-rhqHAguH34TMabufIw5T5dA&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1) [![video](https://img.shields.io/badge/video-YouTube-red)](https://www.youtube.com/watch?v=4Mu5rgfUwoE)

More details are presented in the following papers, [Video](https://www.youtube.com/watch?v=4Mu5rgfUwoE), and [Slides](https://drive.google.com/file/d/1pWzuMKeXzwozWLsFPUuOCRv1JYvT-KXy/view): 

If you find our work is useful in your research, please consider citing:

(1) [Usformer: A Light Neural Network for Left Atrium Segmentation of 3D LGE MRI](https://ieeexplore.ieee.org/abstract/document/10289839)
```bash
@inproceedings{lin2023usformer,
  title={Usformer: A Light Neural Network for Left Atrium Segmentation of 3D LGE MRI},
  author={Lin, Hui and Tapia, Santiago Lopez and Schiffers, Florian and Wu, Yunan and Yang, Huili and Iakovlev, Nikolay and Allen, Bradley D and Avery, Ryan and Lee, Daniel C and Kim, Daniel and others},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)},
  pages={995--999},
  year={2023},
  organization={IEEE}
}





stage 0 preparing data for training i2i model:
       
        python "/home/hln0895/DRL-STNet/stage0.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"

stage 1: train i2i model
	
	python "/home/hln0895/DRL-STNet/stage1.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"

stage 2: generate fake target scans (MRI)

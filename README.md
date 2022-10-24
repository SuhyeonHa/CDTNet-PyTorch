<base target="_blank"/>


# CDTNet-High-Resolution-Image-Harmonization-via-Collaborative-Dual-Transformations (CVPR 2022)

Unofficial implementation of "High-Resolution Image Harmonization via Collaborative Dual Transformations (CVPR 2022)" in PyTorch.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Datasets
- [HAdobe5k](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4)

## Before Training
1. Download HRNet-W18-C model(hrnetv2_w18_imagenet_pretrained.pth) in [HRNets](https://github.com/HRNet/HRNet-Image-Classification)
2. Put it in the `pretrained` folder

## **Base Model**

- Train
```bash 
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train.py --model iih_base --name iih_base_allidh_test --dataset_root ~/IHD/ --dataset_name HAdobe5k --batch_size 80 --init_port 50000
```

- Test
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python test.py --model iih_base --name iih_base_allidh_test --dataset_root ~/IHD/ --dataset_name HAdobe5k --batch_size 80 --init_port 50000
```

- Apply pre-trained model

Download pre-trained model from [Link](url), and put `latest_net_G.pth` in the directory `checkpoints/iih_base_lt_allihd`. Run:
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python test.py --model iih_base --name iih_base_allidh_test --dataset_root ~/IHD/ --dataset_name HAdobe5k --batch_size 80 --init_port 50000
```

# Acknowledgement
We borrowed some of the data modules and model functions from repo of [IntrinsicHarmony](https://github.com/zhenglab/IntrinsicHarmony), [iSSAM](https://github.com/saic-vul/image_harmonization), and [3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

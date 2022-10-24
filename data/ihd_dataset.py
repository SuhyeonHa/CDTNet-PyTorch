"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from util import util
from skimage import color
import cv2
import math

class IhdDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, train_data):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--train_data', type=bool, default=True, help='whether using the training data')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.train_data = opt.train_data
        # self.image_size = opt.crop_size
        self.hr_size = opt.hr_size
        self.lr_size = opt.lr_size
        
        if opt.train_data==True:
            print('loading training file')
            self.trainfile = opt.dataset_root+opt.dataset_name+'_train.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        elif opt.train_data==False:
            #self.real_ext='.jpg'
            print('loading test file')
            # self.trainfile = opt.dataset_root+opt.dataset_name+'_test.txt'
            self.trainfile = opt.dataset_root+opt.dataset_name+'_test.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            # transforms.Normalize((.485, .456, .406), (.229, .224, .225)) # mean, std
        ]
        self.transforms = transforms.Compose(transform_list)
        self.color_transforms = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.2)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        name_parts=path.split('_')
        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_images','real_images')
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')
        # real_g = Image.open(target_path).convert('1')
        # T = transforms.RandomResizedCrop(size = self.lr_size)

        if np.random.rand() > 0.5 and self.train_data:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)
            
        if np.random.rand() > 0.5 and self.train_data:
            crop = transforms.RandomResizedCrop(self.hr_size)
            params = crop.get_params(comp, scale=(0.5, 1.0), ratio=(0.9, 1.1))
            comp = transforms.functional.crop(comp, *params)
            mask = transforms.functional.crop(mask, *params)
            real = transforms.functional.crop(real, *params)

        if comp.size[0] != self.hr_size:
            comp_hr = tf.resize(comp, [self.hr_size, self.hr_size])
            mask_hr = tf.resize(mask, [self.hr_size, self.hr_size])
            real_hr = tf.resize(real, [self.hr_size, self.hr_size])
            # real_g = tf.resize(real_g, [self.image_size, self.image_size])
        
        if comp.size[0] != self.lr_size:
            comp_lr = tf.resize(comp, [self.lr_size, self.lr_size])
            mask_lr = tf.resize(mask, [self.lr_size, self.lr_size])
            real_lr = tf.resize(real, [self.lr_size, self.lr_size])
        
        comp_hr = self.transforms(comp_hr)
        mask_hr = tf.to_tensor(mask_hr)
        real_hr = self.transforms(real_hr)
        
        comp_lr = self.transforms(comp_lr)
        mask_lr = tf.to_tensor(mask_lr)
        real_lr = self.transforms(real_lr)
        
        return {'comp_hr': comp_hr, 'real_hr': real_hr, 'mask_hr':mask_hr, 'comp_lr': comp_lr, 'real_lr': real_lr, 'mask_lr':mask_lr, 'img_path':path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
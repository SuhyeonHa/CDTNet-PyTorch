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
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
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
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        
        if opt.isTrain==True:
            print('loading training file')
            self.trainfile = opt.dataset_root+opt.dataset_name+'_train.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        elif opt.isTrain==False:
            #self.real_ext='.jpg'
            print('loading test file')
            self.trainfile = opt.dataset_root+opt.dataset_name+'_test.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)

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

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)

        if comp.size[0] != self.image_size:
            comp = tf.resize(comp, [self.image_size, self.image_size])
            mask = tf.resize(mask, [self.image_size, self.image_size])
            real = tf.resize(real, [self.image_size, self.image_size])
            # real_g = tf.resize(real_g, [self.image_size, self.image_size])
            
        # ret = transforms.RandomAffine.get_params(degrees=(-10, 10), translate=(0.3, 0.3), scale_ranges=(0.7, 1.2), shears=(-10, 10), img_size=[self.image_size, self.image_size])
        # src_img, src_mask = tf.affine(comp, *ret), tf.affine(mask, *ret)
                
        # center_f = (self.image_size * 0.5 + 0.5, self.image_size * 0.5 + 0.5)
        # translate_f = [1.0 * t for t in ret[1]]
        # affine_mat = self.get_affine_matrix(center_f, ret[0], translate_f, ret[2], ret[3]) # T
        # field_GT = F.affine_grid(affine_mat, self.image_size)
            
        # ncomp = comp
        # outer_mask = mask.copy()
        # # if np.random.rand() > 0.5 and self.isTrain: # apply dilation
        # for _ in range(np.random.randint(10, 21)):
        #     outer_mask = outer_mask.filter(ImageFilter.MaxFilter(3))
        # else:
        #     for _ in range(np.random.randint(6)):
        #         nmask = nmask.filter(ImageFilter.MaxFilter(3)) #dilation

        # inputs=torch.cat([comp,mask],0)
        
        ##RGB2LAB
        # comp = color.rgb2lab(np.array(comp, dtype=np.float32)/255.) #256,256,3 #rgb2lab input: [0,1]
        # real = color.rgb2lab(np.array(real, dtype=np.float32)/255.)
        # src_img = color.rgb2lab(np.array(src_img, dtype=np.float32)/255.)
        ###
        
        # comp = cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2LAB)
        # real = cv2.cvtColor(np.array(real), cv2.COLOR_RGB2LAB)
        
        # comp[:,:,0] = comp[:,:,0]*255/100
        # comp[:,:,1] += 128
        # comp[:,:,2] += 128
        
        # real[:,:,0] = real[:,:,0]*255/100
        # real[:,:,1] += 128
        # real[:,:,2] += 128
        
        # comp /= [100, 127, 127]
        # real /= [100, 127, 127]
        
        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        real = self.transforms(real)
        # outer_mask = tf.to_tensor(outer_mask)
        # src_img = self.transforms(src_img)
        # src_mask = tf.to_tensor(src_mask)
        # affine_mat = torch.Tensor(affine_mat)
        
        # comp = comp * 2.0 - 1.0
        # real = real * 2.0 - 1.0
        # src_img = src_img * 2.0 - 1.0
        
        # real_g = tf.to_tensor(real_g)
        
        ##RGB2LAB
        # comp_L = comp[[0], ...] / 50.0 - 1.0
        # comp_AB = comp[[1, 2], ...] / 110.0
        
        # real_L = real[[0], ...] / 50.0 - 1.0
        # real_AB = real[[1, 2], ...] / 110.0
        
        # src_img_L = src_img[[0], ...] / 50.0 - 1.0
        # src_img_AB = src_img[[1, 2], ...] / 110.0
        ###
        # ncomp_L = comp_L * outer_mask
        # ncomp_AB = comp_AB * outer_mask
        
        return {'comp': comp, 'real': real, 'mask':mask}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
    
    def get_affine_matrix(self, center, angle, translate, scale, shear):
    # center: List[float],
    # angle: float,
    # translate: List[float],
    # scale: float,
    # shear: List[float],) -> List[float]:
        # Helper method to compute inverse matrix for affine transformation

        # Pillow requires inverse affine transformation matrix:
        # Affine matrix is : M = T * C * RotateScaleShear * C^-1
        #
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RotateScaleShear is rotation with scale and shear matrix
        #
        #       RotateScaleShear(a, s, (sx, sy)) =
        #       = R(a) * S(s) * SHy(sy) * SHx(sx)
        #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
        #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
        #         [ 0                    , 0                                      , 1 ]
        # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
        # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
        #          [0, 1      ]              [-tan(s), 1]
        #
        # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

        rot = math.radians(angle)
        sx = math.radians(shear[0])
        sy = math.radians(shear[1])

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        # # Inverted rotation matrix with scale and shear
        # # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        # matrix = [d, -b, 0.0, -c, a, 0.0]
        # matrix = [x / scale for x in matrix]

        # # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        # matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        # matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

        # # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        # matrix[2] += cx
        # matrix[5] += cy
        
        # matrix[2] /= 255 # image_size-1, normalize into (-1, 1)
        # matrix[5] /= 255
        
        # #################
        aff = [a, b, 0.0, c, d, 0.0]
        aff = [x * scale for x in aff]
        
        aff[2] += (cx + tx)
        aff[5] += (cy + ty)
        
        aff[2] -= cx
        aff[5] -= cy

        aff[2] /= 255 # image_size-1, normalize into (-1, 1)
        aff[5] /= 255
        # #################

        return np.array(aff)

    # def masks_to_boxes(self, mask: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute the bounding box according to a given mask.

    #     Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    #     ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    #     Args:
    #         masks (Tensor[N, H, W]): masks to transform where N is the number of masks
    #             and (H, W) are the spatial dimensions.

    #     Returns:
    #         Tensor[N, 4]: bounding boxes
    #     """
    #     # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     #     _log_api_usage_once(masks_to_boxes)
    #     if mask.numel() == 0:
    #         return torch.zeros((0, 4), device=mask.device, dtype=torch.float)

    #     h, w = mask.shape

    #     bounding_box = torch.zeros(4, device=mask.device, dtype=torch.int)
    #     nmask = torch.zeros((h,w), device=mask.device, dtype=torch.float)

    #     y, x = torch.where(mask != 0)
    #     # nmask = torch.zeros((h,w), device=masks.device, dtype=torch.bool)
    #     # if y.numel() == 0:
    #     #     y = torch.zeros(1, device=masks.device, dtype=torch.float)
        
    #     # if x.numel() == 0:
    #     #     x = torch.zeros(1, device=masks.device, dtype=torch.float)

    #     bounding_box[0] = torch.min(x)
    #     bounding_box[1] = torch.min(y)
    #     bounding_box[2] = torch.max(x)
    #     bounding_box[3] = torch.max(y)
        
    #     bounding_box[0] = bounding_box[0].clamp(0, w)
    #     bounding_box[1] = bounding_box[1].clamp(0, h)
    #     bounding_box[2] = bounding_box[2].clamp(0, w)
    #     bounding_box[3] = bounding_box[3].clamp(0, h)
        
    #     nmask[bounding_box[1]:bounding_box[3],bounding_box[0]:bounding_box[2]] = 1.

    #     return nmask.unsqueeze(0)

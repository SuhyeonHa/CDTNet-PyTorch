import torch
import os
import itertools
import torch.nn.functional as F
from util import distributed as du
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
import torch.nn as nn
from torchvision import models
from itertools import chain

from models.hrnet import HRNetIHModel
from models.dih_model import DeepImageHarmonization
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

class IIHBaseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='base', dataset_mode='ihd')
        if is_train:
            parser.add_argument('--lambda_pix', type=float, default=1., help='weight for L1 loss')
            parser.add_argument('--lambda_rgb', type=float, default=1., help='weight for L1 loss')
            parser.add_argument('--lambda_ref', type=float, default=1., help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['G', 'G_pix', 'G_rgb', 'G_ref']
        self.visual_names = ['mask_lr', 'comp_lr', 'out_lr_pix', 'real_lr', 'comp_hr', 'out_hr_rgb', 'out_hr', 'real_hr']
        self.test_visual_names = ['mask_lr', 'comp_lr', 'out_lr_pix', 'real_lr', 'comp_hr', 'out_hr_rgb', 'out_hr', 'real_hr']
        self.model_names = ['G', 'P2P']
        self.opt.device = self.device
        bbparams = {'model': DeepImageHarmonization, 'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True}}
        self.netP2P = HRNetIHModel(base_config = bbparams, small=False)
        self.netP2P.backbone.load_pretrained_weights(opt.hrnet_path)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.netG, opt.init_type, opt.init_gain, opt)
        self.cur_device = torch.cuda.current_device()
        self.netP2P.cuda(self.cur_device)
        self.netG.cuda(self.cur_device)
        self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        self.isTrain = self.opt.isTrain
        if self.ismaster:
            print(self.netG) 
        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netP2P)+str(self.netG))
            self.criterionL1 = torch.nn.L1Loss().cuda(self.cur_device)
            self.criterionL2 = torch.nn.MSELoss().cuda(self.cur_device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(chain(self.netP2P.parameters(), self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer)
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32)
        self.denormalizator = Normalize((-self.mean / self.std), (1.0 / self.std))
        self.normalizator = Normalize(self.mean, self.std)

    def set_input(self, input):
        self.comp_hr = input['comp_hr'].to(self.device)
        self.real_hr = input['real_hr'].to(self.device)
        self.mask_hr = input['mask_hr'].to(self.device)
        self.comp_lr = input['comp_lr'].to(self.device)
        self.real_lr = input['real_lr'].to(self.device)
        self.mask_lr = input['mask_lr'].to(self.device)
        self.image_paths = input['img_path']
        
    def forward(self):
        self.out_lr_pix, F_map, B_map, F_dec = self.netP2P(self.normalizator(self.comp_lr), self.mask_lr)
        self.out_lr_pix = self.denormalizator(self.out_lr_pix)
        
        self.out_hr_rgb, self.out_hr = self.netG(self.out_lr_pix, F_map, B_map, F_dec, self.comp_hr, self.mask_hr, self.opt.train_data)

            

    def backward(self):
        self.loss_G_pix = self.criterionL1(self.out_lr_pix, self.real_lr) * self.opt.lambda_pix
        self.loss_G_rgb = self.criterionL1(self.out_hr_rgb, self.real_hr) * self.opt.lambda_rgb
        self.loss_G_ref = self.criterionL1(self.out_hr, self.real_hr) * self.opt.lambda_ref

        self.loss_G = self.loss_G_pix + self.loss_G_rgb + self.loss_G_ref
        self.loss_G.backward()
        
    def optimize_parameters(self):
            
        # for param in self.netG.parameters():
        #     param.requires_grad = True
        # for name, param in self.netG.named_parameters():
        #     if param.grad == None:
        #         print(name)
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
def load_weights(model, path_to_weights):

    current_state_dict = model.state_dict()
    new_state_dict = torch.load(str(path_to_weights), map_location='cpu')
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
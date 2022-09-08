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

class IIHBaseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='base', dataset_mode='ihd')
        if is_train:
            parser.add_argument('--lambda_real', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_comp', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_comp2real', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_sp', type=float, default=100.0, help='weight for L1 loss')
            # parser.add_argument('--lambda_cr', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['G', 'G_real', 'G_comp', 'G_comp2real', 'G_sp']            
        self.visual_names = ['mask', 'comp', 'real', 'out_fr_br', 'out_fc_bc', 'out_bc_bc']
        self.test_visual_names = ['mask', 'comp', 'out', 'real']
        self.model_names = ['G']
        self.opt.device = self.device
        # self.netE = networks.define_E(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.netG, opt.init_type, opt.init_gain, opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.netG, opt.init_type, opt.init_gain, opt)
        self.cur_device = torch.cuda.current_device()
        self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        if self.ismaster:
            print(self.netG)  
        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            self.criterionL1 = torch.nn.L1Loss().cuda(self.cur_device)
            self.criterionL2 = torch.nn.MSELoss().cuda(self.cur_device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)


    def set_input(self, input):
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        # self.image_paths = input['img_path']
        
    def forward(self):
        if self.isTrain:
            self.out_fr_br, self.out_fc_bc, self.out_bc_bc, self.sp_1, self.sp_2 = self.netG(self.real, self.comp, self.mask)
        else:
            sp, gl_fg, gl_bg = self.netG.module.encoder(self.comp, self.mask)
            self.out = self.netG.module.generator(sp, gl_bg, gl_bg, self.mask)

    def backward(self):
        self.loss_G_real = self.criterionL1(self.out_fr_br, self.real)*self.opt.lambda_real
        self.loss_G_comp = self.criterionL1(self.out_fc_bc, self.comp)*self.opt.lambda_comp
        self.loss_G_comp2real = self.criterionL1(self.out_bc_bc, self.real)*self.opt.lambda_comp2real
        self.loss_G_sp = self.criterionL1(self.sp_1, self.sp_2)*self.opt.lambda_sp
        
        # self.loss_G_transform = self.criterionL1(self.grid_GT, self.gen_grid)*self.opt.lambda_transform
        self.loss_G = self.loss_G_real + self.loss_G_comp + self.loss_G_comp2real + self.loss_G_sp
        self.loss_G.backward()
        # self.loss_G = self.loss_G_first + self.loss_G_second + self.loss_G_transform# + self.loss_G_L_hole #+ self.loss_G_cosine
        # self.loss_G.backward()
        
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


    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y
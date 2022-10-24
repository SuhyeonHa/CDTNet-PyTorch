from glob import glob0
from gzip import BadGzipFile
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from util import util
from . import networks as networks_init
import torchvision.transforms.functional as tff
import math
from models.models_3DLUT import *
from models.basic_blocks import ConvBlock

###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, norm='batch', netG='base', init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'base':
        net = BaseGenerator(input_nc, output_nc, ngf, norm, opt)
        
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)

    return net

class BaseGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm, opt):
        super(BaseGenerator, self).__init__()
        self.device = opt.device
        
        self.LUT0 = Generator3DLUT_identity(dim=64)
        self.LUT1 = Generator3DLUT_zero(dim=64)
        self.LUT2 = Generator3DLUT_zero(dim=64)
        self.LUT3 = Generator3DLUT_zero(dim=64)
        
        self.linear_f = torch.nn.Linear(256*8*8, 256, bias=True)
        self.linear_b = torch.nn.Linear(256*8*8, 256, bias=True)
        
        self.linear_coef = torch.nn.Linear(512, 4, bias=True)
        
        self.refine0 = ConvBlock(39, 10, kernel_size=3, stride=1, padding=1)
        self.refine1 = ConvBlock(10, 5, kernel_size=3, stride=1, padding=1)
        self.conv_attention = nn.Conv2d(5, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(5, 3, kernel_size=1)
        self.trilinear_ = TrilinearInterpolation()
        
        
    def forward(self, out_lr_pix, F_map, B_map, F_dec, comp_hr, mask_hr, train_data):
        
        f_f = self.linear_f(F_map.view(F_map.shape[0], -1))
        f_b = self.linear_b(B_map.view(B_map.shape[0], -1))

        coef = self.linear_coef(torch.cat((f_f, f_b), dim=1))

        new_img = (comp_hr*mask_hr).permute(1,0,2,3).contiguous()
        
        gen_A0 = self.LUT0(new_img)
        gen_A1 = self.LUT1(new_img)
        gen_A2 = self.LUT2(new_img)
        gen_A3 = self.LUT3(new_img)
        
        combine_A = new_img.new(new_img.size())
        for b in range(new_img.size(1)):
            combine_A[:,b,:,:] = coef[b,0] * gen_A0[:,b,:,:] + coef[b,1] * gen_A1[:,b,:,:] + coef[b,2] * gen_A2[:,b,:,:] + coef[b,3] * gen_A3[:,b,:,:] #+ pred[b,4] * gen_A4[:,b,:,:]
        
        if not train_data:
            _, combine_A = self.trilinear_(combine_A, new_img)
            
        out_hr_rgb = combine_A.permute(1,0,2,3) + comp_hr*(1-mask_hr)  #get the [batch_size,3,width,height] combined image
        out_lr_pix = F.interpolate(out_lr_pix, size=comp_hr.size(2))
        F_dec = F.interpolate(F_dec, size=comp_hr.size(2))
        
        out_hr = torch.cat((out_lr_pix, out_hr_rgb, mask_hr, F_dec), dim=1) #3+3+1+32
        out_hr = self.refine0(out_hr)
        out_hr = self.refine1(out_hr)
        attention_map = torch.sigmoid(3.0 * self.conv_attention(out_hr))
        out_hr = attention_map * comp_hr + (1.0 - attention_map) * self.to_rgb(out_hr)
                  
        return out_hr_rgb, out_hr

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

# from models.stylegan2_layers import ConvLayer, ToRGB, EqualLinear, StyledConv
# from models.stylegan2_layers import ResBlock, ConvLayer, ToRGB, EqualLinear, Blur, Upsample, make_kernel
# from models.stylegan2_op import upfirdn2d


###############################################################################
# Helper Functions
###############################################################################

def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

def exists(val):
    return val is not None

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

def define_G(input_nc, output_nc, ngf, norm='instance', netG='base', init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'base':
        net = BaseGenerator(input_nc, output_nc, ngf, norm, opt)
        
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)

    return net

# def define_E(input_nc, output_nc, ngf, norm='instance', netG='base',init_type='normal', init_gain=0.02, opt=None):
#     """Create a generator
#     """
    
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)
    
#     if netG == 'base':
#         net = MixEncoder(input_nc, output_nc, ngf, norm_layer = norm_layer)
        
#     net = networks_init.init_weights(net, init_type, init_gain)
#     net = networks_init.build_model(opt, net)

    return net

class BaseGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm, opt):
        super(BaseGenerator, self).__init__()
        self.device = opt.device
        norm_layer = get_norm_layer(norm_type=norm)

        # self.encoder = MixEncoder(input_nc, output_nc, ngf, norm_layer = norm_layer)
        
        self.reflectance_dim = 256
        self.encoder = ContentEncoder(opt.n_downsample, opt.enc_n_res, opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, opt.pad_type,
                                      opt.spatial_code_ch, opt.global_code_ch)
        # self.generator = MixDecoder(input_nc, output_nc, ngf, norm_layer = norm_layer)
        self.generator = ContentDecoder(opt.n_downsample, opt.dec_n_res, self.encoder.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, opt.pad_type,
                                        opt.spatial_code_ch, opt.global_code_ch)

        
    def forward(self, real, comp, mask):
        
        sp1, fg_gl_1, bg_gl_1 = self.encoder(real, mask)
        sp2, fg_gl_2, bg_gl_2 = self.encoder(comp, mask)
        
        out1 = self.generator(sp1, fg_gl_1, bg_gl_1, mask) #real
        out2 = self.generator(sp2, fg_gl_2, bg_gl_2, mask) #comp
        out3 = self.generator(sp2, bg_gl_2, bg_gl_2, mask) #real

        return out1, out2, out3, sp1, sp2

##############################################################################
# Classes
##############################################################################
  

class BaseLTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseLTGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 0
        i_dec_n_res = 0

        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=pad_type)
       
        self.lighting = GlobalLighting(opt.n_downsample, opt.input_nc+1, opt.ngf, opt.light_mlp_dim, 'none', opt.activ, pad_type=pad_type)
        self.lightingRes = LightingResBlocks(opt.illumination_n_res, self.illumination_enc.output_dim, opt.light_mlp_dim, norm='ln', activation=opt.activ, pad_type=pad_type)
        

    def forward(self, inputs, mask=None, mask_r=None):
        
        l_fg, l_bg = self.lighting(inputs, mask_r)

        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)

        reflectance = self.reflectance_dec(r_content)
        reflectance = reflectance / 2 +0.5

        i_content = self.lightingRes(i_content, l_fg, l_bg, mask_r)
        illumination = self.illumination_dec(i_content)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination

class BaseGDGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseGDGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 4
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=pad_type)

        self.ifm = InharmonyfreeCAModule(opt.n_downsample+1, opt.ifm_n_res, opt.input_nc+1, self.reflectance_enc.output_dim, opt.ngf//2, opt.inharmonyfree_norm, opt.activ, pad_type=pad_type)

        self.reflectanceRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.reflectance_enc.output_dim)
        self.illuminationRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.illumination_enc.output_dim)


    def forward(self, inputs, mask_r, mask_r_32=None):

        match_score, ifm_mean = self.ifm(inputs, mask_r_32)

        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)

        r_content = self.reflectanceRec(r_content, fg_mask=mask_r, attScore=match_score)
        i_content = self.illuminationRec(i_content, fg_mask=mask_r, attScore=match_score.detach())

        reflectance = self.reflectance_dec(r_content)
        reflectance = reflectance / 2 +0.5

        illumination = self.illumination_dec(i_content)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination, ifm_mean

class BaseLTGDGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseLTGDGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 0
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=pad_type)
       
        self.lighting = GlobalLighting(opt.n_downsample, opt.input_nc+1, opt.ngf, opt.light_mlp_dim, 'none', opt.activ, pad_type=pad_type)
        self.lightingRes = LightingResBlocks(opt.illumination_n_res, self.illumination_enc.output_dim, opt.light_mlp_dim, norm='ln', activation=opt.activ, pad_type=pad_type)

        self.ifm = InharmonyfreeCAModule(opt.n_downsample+1, opt.ifm_n_res, opt.input_nc+1, self.reflectance_enc.output_dim, opt.ngf//2, opt.inharmonyfree_norm, opt.activ, pad_type=pad_type)

        self.reflectanceRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.reflectance_enc.output_dim)
        self.illuminationRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.illumination_enc.output_dim)
    def forward(self, inputs, mask_r=None, mask_r_32=None):
        fg_pooling, bg_pooling= self.lighting(inputs, mask_r)
        # match_score, ifm_mean = self.ifm(inputs, mask_r_32, self.lamda)
        match_score, ifm_mean = self.ifm(inputs, mask_r_32)

        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)
        
        r_content = self.reflectanceRec(r_content, fg_mask=mask_r, attScore=match_score)
        
        reflectance = self.reflectance_dec(r_content)
        reflectance = reflectance / 2 +0.5

        i_content = self.lightingRes(i_content, fg_pooling, bg_pooling, mask_r)
        i_content = self.illuminationRec(i_content, fg_mask=mask_r, attScore=match_score.detach())
        
        illumination = self.illumination_dec(i_content)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination, ifm_mean

      
##################################################################################
# Encoder and Decoders
##################################################################################

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type, spatial_code_ch, global_code_ch):
        super(ContentEncoder, self).__init__()
        # output_dim = 256, dim = 64
        self.ToMidpoint = []
        self.ToMidpoint += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.ToMidpoint += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        self.ToMidpoint += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.ToMidpoint += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.ToMidpoint = nn.Sequential(*self.ToMidpoint)
        
        self.ToSpatialCode = nn.Sequential(
                Conv2dBlock(output_dim, output_dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type),
                Conv2dBlock(output_dim, spatial_code_ch, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)
        )
        
        self.ToGlobalFeat = nn.Sequential(
                Conv2dBlock(output_dim, output_dim*2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
                Conv2dBlock(output_dim*2, output_dim*4, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        
        self.ToGlobalCode = EqualLinear(output_dim*4, global_code_ch)
        self.output_dim = output_dim
        

    def forward(self, img, mask):
        x = torch.cat((img, mask), dim=1)
        
        x = self.ToMidpoint(x)
        sp = self.ToSpatialCode(x)
        
        down_mask = F.interpolate(mask, size=x.size(2)) #1/2** n_downsample
        fg = down_mask*x
        bg = (1-down_mask)*x
        
        fg = self.ToGlobalFeat(fg)
        bg = self.ToGlobalFeat(bg)
        
        fg = fg.mean(dim=(2,3))
        bg = bg.mean(dim=(2,3))
        
        fg_gl = self.ToGlobalCode(fg)
        bg_gl = self.ToGlobalCode(bg)
        
        sp = normalize(sp)
        fg_gl = normalize(fg_gl)
        bg_gl = normalize(bg_gl)
        
        return sp, fg_gl, bg_gl

class ContentDecoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type, spatial_code_ch, global_code_ch):
        super(ContentDecoder, self).__init__()
        
        dim = input_dim
        
        self.IntoFeature = GeneratorModulation(global_code_ch, spatial_code_ch)
        
        self.gen1 = GeneratorBlock(global_code_ch, spatial_code_ch, dim, upsample = True)
        self.gen2 = GeneratorBlock(global_code_ch, dim, dim // 2, upsample = True)
        self.gen3 = GeneratorBlock(global_code_ch, dim // 2, dim // 4, upsample = True)
        self.out = Conv2dBlock(dim // 4, output_dim, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type)
        
        # self.ToRGB = nn.Sequential(
        #     GeneratorBlock(global_code_ch, spatial_code_ch, dim, upsample = True),
        #     GeneratorBlock(global_code_ch, dim, dim // 2, upsample = True),
        #     GeneratorBlock(global_code_ch, dim // 2, dim // 4, upsample = True),
        #     # nn.Upsample(scale_factor=2),
        #     # Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
        #     Conv2dBlock(dim // 4, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        # )
        # self.trans_1 = GeneratorBlock(global_code_ch, spatial_code_ch, dim, upsample = True)
        # self.trans_2 = GeneratorBlock(global_code_ch, dim, dim // 2, upsample = True)
        # self.trans_3 = GeneratorBlock(global_code_ch, dim // 2, dim // 4, upsample = True)
        
        # self.model = []
        
        # upsampling blocks
        # for i in range(n_downsample):
        #     self.model += [
        #         nn.Upsample(scale_factor=2),
        #         Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
        #     ]
        #     dim //= 2

        # self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        # self.model = nn.Sequential(*self.model)

    def forward(self, sp, fl, bl, mask):
        sp = normalize(sp)
        fl = normalize(fl)
        bl = normalize(bl)
        
        x_f = self.IntoFeature(sp, fl)
        x_b = self.IntoFeature(sp, bl)
        
        down_mask = F.interpolate(mask, size=x_f.size(2))
        # fg = down_mask*x_f
        # bg = (1-down_mask)*x_b
        x = down_mask*x_f + (1-down_mask)*x_b
        
        x = self.gen1(x, fl, bl, mask)
        x = self.gen2(x, fl, bl, mask)
        x = self.gen3(x, fl, bl, mask)
        x = self.out(x)
        
        return x

class GlobalLighting(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, light_mlp_dim=8, norm=None, activ=None, pad_type='zero'):
    
        super(GlobalLighting, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        # self.model += [ResBlocks(4, dim, norm=norm, activation=activ, pad_type=pad_type)]
        # if not dim == output_dim:
        #     self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.light_mlp = LinearBlock(dim, light_mlp_dim, norm='none', activation='none')

    def forward(self, x, fg_mask):
        x = self.model(x)
        b,c,h,w = x.size()

        fg_mask_sum = torch.sum(fg_mask.view(b, 1, -1), dim=2)
        bg_mask_sum = h*w - fg_mask_sum+1e-8
        fg_mask_sum = fg_mask_sum +1e-8
        x_bg = x*(1-fg_mask)

        # avg pooling to 1*1
        x_bg_pooling = torch.sum(x_bg.view(b,c,-1), dim=2).div(bg_mask_sum)
        l_bg = self.light_mlp(x_bg_pooling)

        x_fg = x*fg_mask
        x_fg_pooling = torch.sum(x_fg.view(b,c,-1), dim=2).div(fg_mask_sum)
        l_fg = self.light_mlp(x_fg_pooling)

        return l_fg, l_bg
    
class InharmonyfreeCAModule(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type, lamda=10):
        super(InharmonyfreeCAModule, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 3,1,1, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.in_normal = nn.InstanceNorm1d(1024)

    def forward(self, x, fg_mask=None,lamda=10):
        content = self.model(x)
        b,c,h,w = content.size()

        fg = content*fg_mask
        bg = content*(1-fg_mask)
        fg_patch = fg.view(b,c,-1).permute(0,2,1)
        bg_patch = bg.view(b,c,-1)

        fg_patch_mu = torch.mean(fg_patch, dim=2, keepdim=True)
        bg_patch_mu = torch.mean(bg_patch, dim=1, keepdim=True)
        fg_bg_conv = torch.matmul((fg_patch-fg_patch_mu), (bg_patch-bg_patch_mu))/(c-1)
        
        match_score_soft = F.softmax(lamda * fg_bg_conv.permute(0,2,1), dim=1)
        match_score_soft = match_score_soft.view(b, -1, h, w)
        content_mean = torch.mean(content, dim=1, keepdim=True)
        return match_score_soft, content_mean

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class LightingResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, light_mlp_dim=8, norm='in', activation='relu', pad_type='zero'):
        super(LightingResBlocks, self).__init__()
        self.resblocks = nn.ModuleList([LightingResBlock(dim, light_mlp_dim, norm=norm, activation=activation, pad_type=pad_type) for i in range(num_blocks)])

    def forward(self, x, fg, bg, fg_mask):
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, fg, bg, fg_mask)

        return x

class HarmonyRecBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(HarmonyRecBlocks, self).__init__()
        self.resblocks = nn.ModuleList([HarmonyRecBlock(stride=1, rate=2, channels=dim) for i in range(num_blocks)])

    def forward(self, x, fg_mask=None, attScore=None):
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, fg_mask, attScore)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, input_dim//2, norm=norm, activation=activ)]
        dim = input_dim//2
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim//2, norm=norm, activation=activ)]
            dim = dim//2
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class LightingResBlock(nn.Module):
    def __init__(self, dim, light_mlp_dim, norm='in', activation='relu', pad_type='zero'):
        super(LightingResBlock, self).__init__()
        
        self.lt_1 = Lighting(dim, light_mlp_dim, norm=norm, activation=activation, pad_type=pad_type)
        self.conv_1 = nn.Conv2d(dim, dim, 3, 1)
        self.lt_2 = Lighting(dim, light_mlp_dim, norm=norm, activation=activation, pad_type=pad_type)
        self.conv_2 = nn.Conv2d(dim, dim, 3, 1)
        self.norm = LayerNorm(dim)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x, fg, bg, fg_mask):
        residual = x
        out_1 = self.actvn(self.norm(self.lt_1(self.conv_1(self.pad(x)), fg, bg, fg_mask)))
        out = self.norm(self.lt_2(self.conv_2(self.pad(out_1)), fg, bg, fg_mask))
        out += residual
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class Lighting(nn.Module):
    def __init__(self, dim, light_mlp_dim, norm='ln', activation='relu', pad_type='zero'):
        super(Lighting, self).__init__()

        self.light_mlp = LinearBlock(light_mlp_dim, 4*dim, norm='none', activation=activation)
        self.rgb_model = Conv2dBlock(dim ,dim*3, 3, 1, 1, norm='grp', activation=activation, pad_type=pad_type, groupcount=3)

        self.dim = dim

    def forward(self, x, fg, bg, fg_mask):
        residual = x
        b,c,h,w = x.size()
        illu_fg_color_ratio, illu_fg_intensity = self.illumination_extract(fg)
        illu_bg_color_ratio, illu_bg_intensity = self.illumination_extract(bg)
        
        illu_color_ratio = illu_bg_color_ratio.div(illu_fg_color_ratio+1e-8)
        illu_intensity = illu_bg_intensity - illu_fg_intensity
        b,c,h,w = x.size()

        x_rgb = self.rgb_model(x)
        x_rgb = x_rgb.view(b,3,c,h, w)
        illu_color_ratio = illu_color_ratio.view(b, 3, c, 1, 1).expand_as(x_rgb)
        x_t_c = torch.sum(x_rgb*illu_color_ratio, dim=1)

        illu_intensity = illu_intensity.view(b,c,1,1).expand_as(x)

        x_t = x_t_c + illu_intensity
        
        output = residual*(1-fg_mask)+x_t*fg_mask
        return output

    
    def illumination_extract(self, x):
        b = x.size(0)
        illumination = self.light_mlp(x)
        illu_intensity = illumination[:, :self.dim]
        illu_color = illumination[:, self.dim:].view(b, 3, self.dim)
        illu_color_ratio = torch.softmax(illu_color,dim=1)
        return illu_color_ratio, illu_intensity

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class HarmonyRecBlock(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, channels=128):
        super(HarmonyRecBlock, self).__init__()
        self.ksize = ksize
        self.kernel = 2 * rate
        self.stride = stride
        self.rate = rate
        self.harmonyRecConv = Conv2dBlock(channels*2, channels, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')

    def forward(self, bg_in, fg_mask=None, attScore=None):
        b, dims, h, w = bg_in.size()
        
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(bg_in, ksizes=[self.kernel, self.kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(b, dims, self.kernel, self.kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        ACL = []

        for ib in range(b):
            CA = attScore[ib:ib+1, :, :, :]
            k2 = raw_w[ib, :, :, :, :]
            ACLt = F.conv_transpose2d(CA, k2, stride=self.rate, padding=1)
            ACLt = ACLt / 4
            if ib == 0:
                ACL = ACLt
            else:
                ACL = torch.cat([ACL, ACLt], dim=0)
        con1 = torch.cat([bg_in, ACL], dim=1)
        ACL2 = self.harmonyRecConv(con1)
        return ACL2+bg_in

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x
    
#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, latent_channels=48, pad_type='zero', activation='lrelu', norm='in'):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels * 2, 3, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels//2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels//2, out_channels, 3, 1, 1, pad_type = pad_type, activation = 'none', norm = norm),
            nn.Tanh()
      )
        
        self.refine_conv = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels*2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = pad_type, activation = activation, norm = norm)
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels*2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*4, 3, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = 'relu', norm = norm)
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm)
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(latent_channels*8, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*4, latent_channels*4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            TransposeGatedConv2d(latent_channels * 4, latent_channels*2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels*2, latent_channels*2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, latent_channels//2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels//2, out_channels, 3, 1, 1, pad_type = pad_type, activation = 'none', norm = norm),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                           # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        refine_conv = self.refine_conv(second_in)     
        refine_atten = self.refine_atten_1(second_in)
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out
    
class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=True, use_cuda=True, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
            Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L] [4, 192, 4, 4, 1024]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        mask = F.interpolate(mask, scale_factor=1./self.rate, mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')

        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        return y
    
#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
# class GatedGenerator(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, latent_channels=64, pad_type='zero', activation='lrelu', norm='in'):
#         super(GatedGenerator, self, ).__init__()

#         self.coarse = nn.Sequential(
#             # encoder
#             GatedConv2d(in_channels, latent_channels, 7, 1, 3, pad_type = pad_type, activation = activation, norm = 'none'),
#             GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
#             # Bottleneck
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             # decoder
#             TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels, out_channels, 7, 1, 3, pad_type = pad_type, activation = 'tanh', norm = 'none')
#         )
#         self.refinement = nn.Sequential(
#             # encoder
#             GatedConv2d(in_channels, latent_channels, 7, 1, 3, pad_type = pad_type, activation = activation, norm = 'none'),
#             GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
#             # Bottleneck
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             # decoder
#             TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
#             GatedConv2d(latent_channels, out_channels, 7, 1, 3, pad_type = pad_type, activation = 'tanh', norm = 'none')
#         )
        
#     def forward(self, img, in_mask, aligned, aligned_mask):
#         # img: entire img
#         # mask: 1 for mask region; 0 for unmask region
#         # 1 - mask: unmask
#         # img * (1 - mask): ground truth unmask region
#         # Coarse
#         # print(img.shape, mask.shape)
        
#         diff_mask = in_mask - aligned_mask
#         union_mask = in_mask + aligned_mask
#         first_masked_img = img * (1 - union_mask) + aligned * aligned_mask # bg+fg
#         first_in = torch.cat((first_masked_img, diff_mask), 1)       # in: [B, 4, H, W]
#         first_out = self.coarse(first_in)                       # out: [B, 3, H, W]
#         # Refinement
#         second_masked_img = img * (1 - union_mask) + aligned * aligned_mask + first_out * diff_mask
#         # second_masked_img = img * (1 - mask) + first_out * mask
#         second_in = torch.cat((second_masked_img, union_mask), 1)     # in: [B, 4, H, W]
#         second_out = self.refinement(second_in)                 # out: [B, 3, H, W]
#         return first_masked_img, second_masked_img, first_out, second_out

class GatedGeneratorCoarse(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, latent_channels=64, pad_type='zero', activation='lrelu', norm='in'):
        super(GatedGeneratorCoarse, self, ).__init__()

        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 7, 1, 3, pad_type = pad_type, activation = activation, norm = 'none'),
            GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, out_channels, 7, 1, 3, pad_type = pad_type, activation = 'tanh', norm = 'none')
        )
        
    def forward(self, img, in_mask, aligned, aligned_mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
        # print(img.shape, mask.shape)
        diff_mask = in_mask - aligned_mask
        union_mask = in_mask + aligned_mask
        first_masked_img = img * (1 - union_mask) + aligned * aligned_mask # bg+fg
        first_in = torch.cat((first_masked_img, diff_mask), 1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                       # out: [B, 3, H, W]
        return first_masked_img, first_out
    
class GatedGeneratorRefine(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, latent_channels=64, pad_type='zero', activation='lrelu', norm='in'):
        super(GatedGeneratorRefine, self, ).__init__()

        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 7, 1, 3, pad_type = pad_type, activation = activation, norm = 'none'),
            GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm),
            GatedConv2d(latent_channels, out_channels, 7, 1, 3, pad_type = pad_type, activation = 'tanh', norm = 'none')
        )
        
    def forward(self, img):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
        # print(img.shape, mask.shape)
        
        # diff_mask = in_mask - aligned_mask
        # union_mask = in_mask + aligned_mask
        # # Refinement
        # second_masked_img = img * (1 - union_mask) + aligned * aligned_mask + first_out * diff_mask
        # # second_masked_img = img * (1 - mask) + first_out * mask
        # second_in = torch.cat((second_masked_img, union_mask), 1)     # in: [B, 4, H, W]
        output = self.refinement(img)                 # out: [B, 3, H, W]
        return output
##################################################################################
# WEnd-to-end weakly-supervised semantic alignment
##################################################################################
    
def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)



class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]
            
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        if self.shape=='3D':
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        elif self.shape=='4D':
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
        
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, kernel_sizes=[7,5], channels=[256,64], feature_size=16):
        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = feature_size*feature_size
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        # self.linear = nn.Linear(ch_out * k_size * k_size, output_dim)
        # self.linear1 = nn.Linear(30976, 2304)
        self.linear2 = nn.Linear(2304, output_dim)
        if use_cuda:
            self.conv.cuda()
            # self.linear1.cuda()
            self.linear2.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        # x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    
class CNNGeometric(nn.Module):
    def __init__(self, output_dim=6, 
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 return_correlation=False,  
                 fr_feature_size=16,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_self_matching=False,
                 normalize_features=True, normalize_matches=True, 
                 batch_normalization=True, 
                 train_fe=False,use_cuda=True):
#                 regressor_channels_1 = 128,
#                 regressor_channels_2 = 64):
        
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches)        
        

        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda,
                                                   feature_size=fr_feature_size,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)


        self.ReLU = nn.ReLU(inplace=True)
    
    # used only for foward pass at eval and for training with strong supervision
    def forward(self, src_img, tgt_img): 
        # feature extraction
        feature_A = self.FeatureExtraction(src_img)
        feature_B = self.FeatureExtraction(tgt_img)
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        
        if self.return_correlation:
            return (theta,correlation)
        else:
            return theta
    
    
# class CNNGeometric(nn.Module):
#     def __init__(self, use_cuda=True):
        
#         super(CNNGeometric, self).__init__()
#         self.use_cuda = use_cuda
#         self.localization = nn.Sequential(
#             nn.Conv2d(4, 16, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True) # 16*16*64
#         )
        
        
#         self.fc_loc = nn.Sequential(
#             nn.Linear(16 * 16 * 64, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )

#         self.ReLU = nn.ReLU(inplace=True)
#         self.FeatureExtraction = FeatureExtraction()
#         self.FeatureCorrelation = FeatureCorrelation()
        
    
#     # used only for foward pass at eval and for training with strong supervision
#     def forward(self, bg, obj): 
#         # feature extraction
#         bg_feats, obj_feates = self.FeatureExtraction(bg), self.FeatureExtraction(obj)
#         correlation = self.FeatureCorrelation(bg_feats,obj_feates)
#         features = self.localization(correlation) #change to original below.
#         # feature correlation
#         # regression to tnf parameters theta
#         xs = features.view(-1, 16 * 16 * 64)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         return theta
##########################
#https://github.com/researchmm/AOT-GAN-for-Inpainting/
##########################
class InpaintGenerator(nn.Module):
    def __init__(self, rates=[1,2,4,8], block_num=8):  # 1046
        super(InpaintGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        # self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

#############################################
################
# https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/u-net/u_net.py
################

def ConvBlock(in_dim, out_dim, act_fn, norm, kernel_size=3, stride=1, padding=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
        norm(out_dim),
        act_fn,
    )
    return model

def ConvTransBlock(in_dim, out_dim, act_fn, norm, kernel_size=3, stride=1, padding=1):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
        norm(out_dim),
        act_fn,
    )
    return model

def Maxpool():
    pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    return pool

def ConvBlock2X(in_dim, out_dim, act_fn, norm, kernel_size=3, stride=1, padding=1):
    model = nn.Sequential(
        ConvBlock(in_dim, out_dim, act_fn, norm, kernel_size),
        ConvBlock(out_dim, out_dim, act_fn, norm, kernel_size),
    )
    return model

class MixEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter, norm_layer):
        super(MixEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.norm_layer = norm_layer
        act_fn = nn.LeakyReLU(0.2, inplace = True)
        self.spatial_code_ch = 8
        self.global_code_ch = 1024

        self.down_1 = ConvBlock2X(self.in_dim, self.num_filter, act_fn, self.norm_layer)
        self.pool_1 = Maxpool()
        self.down_2 = ConvBlock2X(self.num_filter, self.num_filter * 2, act_fn, self.norm_layer)
        self.pool_2 = Maxpool()
        self.down_3 = ConvBlock2X(self.num_filter * 2, self.num_filter * 4, act_fn, self.norm_layer)
        self.pool_3 = Maxpool()
        # self.down_4 = ConvBlock2X(self.num_filter * 4, self.num_filter * 8, act_fn, self.norm_layer)
        # self.pool_4 = Maxpool()
        ######
        
        self.bridge_1 = ConvBlock2X(self.num_filter * 4, self.num_filter * 4, act_fn, self.norm_layer)
        
        self.ToSpatialCode = nn.Sequential(
            ConvBlock(self.num_filter * 4, self.num_filter * 4, act_fn, self.norm_layer, kernel_size=1, stride=1, padding=0),
            ConvBlock(self.num_filter * 4, self.spatial_code_ch, act_fn, self.norm_layer, kernel_size=1, stride=1, padding=0),
            )
        
        self.ToGlobalCode = nn.Sequential(
            EqualLinear(self.num_filter*16, self.global_code_ch),
        )
        
        self.down_4 = ConvBlock2X(self.num_filter * 4, self.num_filter * 8, act_fn, self.norm_layer, kernel_size=1, stride=1, padding=0)
        self.pool_4 = Maxpool()
        self.down_5 = ConvBlock2X(self.num_filter * 8, self.num_filter * 16, act_fn, self.norm_layer, kernel_size=1, stride=1, padding=0)
        self.pool_5 = Maxpool()
                
    def forward(self, image, mask, return_feats=False):
        input = torch.cat((image, mask), dim=1)
        
        down_1 = self.down_1(input) # concat w/ trans_4
        pool_1 = self.pool_1(down_1) 
        down_2 = self.down_2(pool_1) # concat w/ trans_3
        pool_2 = self.pool_2(down_2) 
        down_3 = self.down_3(pool_2) # concat w/ trans_2
        pool_3 = self.pool_3(down_3) # torch.Size([8, 256, 32, 32])

        pool_3 = self.bridge_1(pool_3)
        
        sp = self.ToSpatialCode(pool_3) #torch.Size([8, 256, 32, 32]) #TODO: shoudlbe changed into ResBlock
        
        down_4 = self.down_4(pool_3) # concat w/ trans_2
        pool_4 = self.pool_4(down_4) 
        down_5 = self.down_5(pool_4) # concat w/ trans_1
        pool_5 = self.pool_5(down_5) 
        
        pool_5 = pool_5.mean(dim=(2,3)) # torch.Size([8, 1024])
        
        gl = self.ToGlobalCode(pool_5) #torch.Size([8, 256])
        
        sp = normalize(sp)
        gl = normalize(gl)
        
        feats = [None, down_3, down_2, down_1]
        
        if return_feats:
            return sp, gl, feats
        else:
            return sp, gl
        
class MixDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter, norm_layer):
        super(MixDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.norm_layer = norm_layer
        act_fn = nn.LeakyReLU(0.2, inplace = True)
        self.spatial_code_ch = 8
        self.global_code_ch = 1024
            
        # self.FromSpatialCode = nn.Sequential(
        #     ConvBlock(self.spatial_code_ch, self.num_filter * 4, act_fn, self.norm_layer, kernel_size=1, stride=1, padding=0),
        #     ConvBlock(self.num_filter * 4, self.num_filter * 4, act_fn, self.norm_layer, kernel_size=1, stride=1, padding=0),
        #     )
        
        # self.FromGlobalCode = nn.Sequential(
        #     EqualLinear(self.global_code_ch, self.num_filter*16)
        # )
        self.IntoFeature = GeneratorModulation(opt.global_code_ch, opt.spatial_code_ch)
        
        self.trans_1 = GeneratorBlock(self.global_code_ch, self.spatial_code_ch, self.num_filter * 4, upsample = True)
        self.trans_2 = GeneratorBlock(self.global_code_ch, self.num_filter * 4, self.num_filter * 2, upsample = True)
        self.trans_3 = GeneratorBlock(self.global_code_ch, self.num_filter * 2, self.num_filter, upsample = True)
        # self.trans_4 = GeneratorBlock(self.global_code_ch, self.num_filter, self.num_filter, upsample = True)
        
        # self.trans_1 = ConvTransBlock(self.num_filter * 16,self.num_filter * 8, act_fn, self.norm_layer)
        self.up_1 = ConvBlock2X(self.num_filter * 8, self.num_filter * 4, act_fn, self.norm_layer)
        # self.trans_2 = ConvTransBlock(self.num_filter * 8, self.num_filter * 4, act_fn, self.norm_layer)
        self.up_2 = ConvBlock2X(self.num_filter * 4, self.num_filter * 2, act_fn, self.norm_layer)
        # self.trans_3 = ConvTransBlock(self.num_filter * 4, self.num_filter * 2, act_fn, self.norm_layer)
        self.up_3 = ConvBlock2X(self.num_filter * 2, self.num_filter, act_fn, self.norm_layer)
        # self.trans_4 = ConvTransBlock(self.num_filter * 2, self.num_filter, act_fn, self.norm_layer)
        # self.up_4 = ConvBlock2X(self.num_filter, self.out_dim, act_fn, self.norm_layer)


        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 3, 1, 1),
            nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace = True),
        )

    def forward(self, sp, gl, feats):
        
        sp = normalize(sp)
        gl = normalize(gl)
        
        # sp = self.FromSpatialCode(sp)
        # gl = self.FromGlobalCode(gl)
        
        x = self.IntoFeature(sp, gl) # x:torch.Size([8, 8, 32, 32])
        
        trans_1 = self.trans_1(x, gl) # torch.Size([8, 256, 32, 32])
        concat_1 = torch.cat([trans_1, feats[1]], dim = 1) # torch.Size([8, 512, 64, 64])
        up_1 = self.up_1(concat_1) # torch.Size([8, 256, 64, 64])
        trans_2 = self.trans_2(up_1, gl) 
        concat_2 = torch.cat([trans_2, feats[2]], dim = 1) # torch.Size([8, 256, 128, 128])
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2, gl)
        concat_3 = torch.cat([trans_3, feats[3]], dim = 1) # torch.Size([8, 128, 256, 256])
        up_3 = self.up_3(concat_3) # torch.Size([8, 64, 256, 256])
        # trans_4 = self.trans_1(up_3, gl)
        # concat_4 = torch.cat([trans_4, feats[4]], dim = 1)
        # up_4 = self.up_1(concat_4, gl)
        out = self.out(up_3)# torch.Size([8, 3, 256, 256])
        return out
        
        
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), nn.LeakyReLU(0.2)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)
        
class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        # self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        # self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, fstyle, bstyle, mask): #inoise
        if exists(self.upsample):
            x = self.upsample(x)
            
        down_mask = F.interpolate(mask, size=x.size(2))

        # inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        # noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        # noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))
        fstyle1 = self.to_style1(fstyle) #torch.Size([8, 8])
        fx = self.conv1(x*down_mask, fstyle1)
        # x = self.activation(x + noise1)
        fx = self.activation(fx)
        
        bstyle1 = self.to_style1(bstyle)
        bx = self.conv1(x*(1-down_mask), bstyle1)
        bx = self.activation(bx)

        fstyle2 = self.to_style2(fstyle) #torch.Size([8, 256])
        fx = self.conv2(fx*down_mask, fstyle2) #
        fx = self.activation(fx)
        
        bstyle2 = self.to_style2(bstyle)
        bx = self.conv2(bx*(1-down_mask), bstyle2)
        bx = self.activation(bx)
        # x = self.activation(x + noise2)

        x = fx*down_mask+bx*(1-down_mask)
        return x
    
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y): # x:spatial y:style
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask, gp=True)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False, use_mixing=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.use_attention = use_attention
        if use_attention:
            attention_conv = nn.Conv2d(outer_nc+input_nc, outer_nc+input_nc, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_sigmoid])

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            ret = torch.cat([x, self.model(x)], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


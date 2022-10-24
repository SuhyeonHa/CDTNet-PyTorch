import torch
import torch.nn as nn

from .conv_autoencoder import ConvEncoder, DeconvDecoder
import torch.nn.functional as F


class DeepImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = DeconvDecoder(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output, F_dec = self.decoder(intermediates, image, mask)
        
        bottleneck_layer = 2 #upside down
        down_mask = F.interpolate(mask, size=intermediates[bottleneck_layer].size(2))
        f_map = down_mask * intermediates[bottleneck_layer]
        b_map = (1-down_mask) * intermediates[bottleneck_layer]
        return output, f_map, b_map, F_dec
        # return {'images': output, 'feats': intermediates, 'F_dec': F_dec}
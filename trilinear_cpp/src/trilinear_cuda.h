#ifndef TRILINEAR_CUDA_H
#define TRILINEAR_CUDA_H

#import <torch/extension.h>

int trilinear_forward_cuda(torch::Tensor &lut, torch::Tensor &image, torch::Tensor &output,
                           const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch);

int trilinear_backward_cuda(torch::Tensor &image, torch::Tensor &image_grad, torch::Tensor &lut_grad,
                            const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch);


#endif

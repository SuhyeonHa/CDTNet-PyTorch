#include "trilinear_kernel.cuh"
#include <torch/extension.h>
// #include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

int trilinear_forward_cuda(torch::Tensor &lut, torch::Tensor &image, torch::Tensor &output,
                           const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch)
{
    // Grab the input tensor
    float * lut_flat = (float *) lut.data_ptr<float>();
    float * image_flat = (float *) image.data_ptr<float>();
    float * output_flat = (float *) output.data_ptr<float>();

    TriLinearForwardLaucher(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

int trilinear_backward_cuda(torch::Tensor &image, torch::Tensor &image_grad, torch::Tensor &lut_grad,
                            const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch)
{
    // Grab the input tensor
    float * image_grad_flat = (float *) image_grad.data_ptr<float>();
    float * image_flat = (float *) image.data_ptr<float>();
    float * lut_grad_flat = (float *) lut_grad.data_ptr<float>();

    TriLinearBackwardLaucher(image_flat, image_grad_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trilinear_forward_cuda, "Trilinear forward");
  m.def("backward", &trilinear_backward_cuda, "Trilinear backward");
}


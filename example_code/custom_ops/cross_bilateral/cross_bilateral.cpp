//  Copyright (c) 2024 CGLab, GIST. All rights reserved.
 
//  Redistribution and use in source and binary forms, with or without modification, 
//  are permitted provided that the following conditions are met:
 
//  - Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.
//  - Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.
 
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> cuda_cross_bilateral_forward(torch::Tensor input, torch::Tensor albedo, torch::Tensor normal, torch::Tensor depth, int winSize);
torch::Tensor cuda_cross_bilateral_backward(torch::Tensor inGrad, torch::Tensor input, torch::Tensor albedo, torch::Tensor normal, torch::Tensor depth, torch::Tensor wgtSum, int winSize);

std::vector<torch::Tensor> cross_bilateral_forward(
    torch::Tensor input,
    torch::Tensor albedo,
    torch::Tensor normal,
    torch::Tensor depth,
    int winSize
) {

    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(normal);
    CHECK_INPUT(albedo);

    return cuda_cross_bilateral_forward(input, albedo, normal, depth, winSize);
}

torch::Tensor cross_bilateral_backward(
    torch::Tensor inGrad,
    torch::Tensor input,
    torch::Tensor albedo,
    torch::Tensor normal,
    torch::Tensor depth,
    torch::Tensor wgtSum,
    int winSize
) {

    CHECK_INPUT(inGrad);
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(normal);
    CHECK_INPUT(albedo);
    CHECK_INPUT(wgtSum);

    return cuda_cross_bilateral_backward(inGrad, input, albedo, normal, depth, wgtSum, winSize);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_bilateral_forward", &cross_bilateral_forward, "cross_bilateral_forward");
    m.def("cross_bilateral_backward", &cross_bilateral_backward, "cross_bilateral_backward");
}

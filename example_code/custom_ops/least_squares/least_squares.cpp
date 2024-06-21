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

torch::Tensor cuda_ols_forward(torch::Tensor input, torch::Tensor depth, torch::Tensor albedo, torch::Tensor normal, int winSize);
torch::Tensor cuda_ols_backward(torch::Tensor inGrad, torch::Tensor input, torch::Tensor depth, torch::Tensor albedo, torch::Tensor normal, int winSize);


torch::Tensor least_squares_forward(
    torch::Tensor input,
    torch::Tensor depth,
    torch::Tensor albedo,
    torch::Tensor normal, 
    int win_size
) {
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(albedo);
    CHECK_INPUT(normal);

    return cuda_ols_forward(input, depth, albedo, normal, win_size);
}

torch::Tensor least_squares_backward(
    torch::Tensor inGrad,
    torch::Tensor input,
    torch::Tensor depth,
    torch::Tensor albedo,
    torch::Tensor normal,
    int win_size
) {

    CHECK_INPUT(inGrad);
    CHECK_INPUT(input);
    CHECK_INPUT(depth);
    CHECK_INPUT(albedo);
    CHECK_INPUT(normal);

    return cuda_ols_backward(inGrad, input, depth, albedo, normal, win_size);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("least_squares_forward", &least_squares_forward, "least_squares_forward");
    m.def("least_squares_backward", &least_squares_backward, "least_squares_backward");
}

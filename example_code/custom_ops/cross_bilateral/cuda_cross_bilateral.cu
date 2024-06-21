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
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

__forceinline__ __host__ __device__ int iDivUp(int a, int b) { 
      return ((a % b) != 0) ? (a / b + 1) : (a / b); 
}

#define FLT_EPS 0.0001f

__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __device__ float Dot(const float4& a, const float4& b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float a, float4 b) {
    return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}
__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x) + (a.y * a.y) + (a.z * a.z);
}



__global__ void CrossBilateralFilterKernel(const float* _img, const float* _albedo, const float* _normal, const float* _depth,
							 float* _output, float*_outWgtSum, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;


	float bandwidth =  0.1f;
	float bandwidth_alb =  0.05f;
	float bandwidth_pos = (float)(halfWinSize / 3.f);

	const float4& cAlbedo = make_float4(_albedo[cIdx * 3 + 0], _albedo[cIdx * 3 + 1], _albedo[cIdx * 3 + 2], 0.f);
	const float4& cNormal = make_float4(_normal[cIdx * 3 + 0], _normal[cIdx * 3 + 1], _normal[cIdx * 3 + 2], 0.f);
	const float& cDepth = _depth[cIdx];

    float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);
    float totalWgt = 0.f;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;

			const float4& iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
			const float4& iImg = make_float4(_img[idx * 3 + 0], _img[idx * 3 + 1], _img[idx * 3 + 2], 0.f);
			const float& iDepth = _depth[idx];
			
            float dist2_albedo = norm2(cAlbedo - iAlbedo) / (2.f * bandwidth_alb * bandwidth_alb);
			float dist2_normal = norm2(cNormal - iNormal) / (2.f * bandwidth * bandwidth);
			float dist2_pos = ((cx-x)*(cx-x) + (cy-y)*(cy-y)) / (2.f * bandwidth_pos * bandwidth_pos);
			float dist2_depth = ((cDepth - iDepth) * (cDepth - iDepth)) / (2.f * bandwidth * bandwidth);
			float total_dist = (dist2_albedo + dist2_normal + dist2_pos + dist2_depth);
			float wgt = __expf(-total_dist);

            accCol += iImg * wgt;
            totalWgt += wgt;

		}
	}

    float invWgt = 1.f / fmaxf(FLT_EPS, totalWgt);
	
	_outWgtSum[cIdx] = totalWgt;
    _output[cIdx * 3 + 0] = accCol.x * invWgt;
	_output[cIdx * 3 + 1] = accCol.y * invWgt;
    _output[cIdx * 3 + 2] = accCol.z * invWgt;
}


__global__ void CrossBilateralFilterBackwardKernel(const float* _inGrad, const float* _img, const float* _albedo, const float* _normal, const float* _depth, const float* _wgtSum,
							 float* _outGrad, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;

	const float4& cAlbedo = make_float4(_albedo[cIdx * 3 + 0], _albedo[cIdx * 3 + 1], _albedo[cIdx * 3 + 2], 0.f);
	const float4& cNormal = make_float4(_normal[cIdx * 3 + 0], _normal[cIdx * 3 + 1], _normal[cIdx * 3 + 2], 0.f);
	const float& cDepth = _depth[cIdx];

    float bandwidth =  0.1f;
	float bandwidth_alb =  0.05f;
	float bandwidth_pos = (float)(halfWinSize / 3.f);

    float4 accGrad = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;

			const float4& iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
			const float& iDepth = _depth[idx];

			float dist2_albedo = norm2(cAlbedo - iAlbedo) / (2.f * bandwidth_alb * bandwidth_alb);
			float dist2_normal = norm2(cNormal - iNormal) / (2.f * bandwidth * bandwidth);
			float dist2_pos = ((cx-x)*(cx-x) + (cy-y)*(cy-y)) / (2.f * bandwidth_pos * bandwidth_pos);
			float dist2_depth = ((cDepth - iDepth) * (cDepth - iDepth)) / (2.f * bandwidth * bandwidth);
			

            float total_dist;
			total_dist = (dist2_albedo + dist2_normal + dist2_pos + dist2_depth);
			float wgt = __expf(-total_dist);

			const float& wgtSum = _wgtSum[idx];
			float invWgt = 1.f / fmaxf(FLT_EPS, wgtSum);
			wgt = wgt * invWgt;

            const float4& inGrad = make_float4(_inGrad[idx * 3 + 0], _inGrad[idx * 3 + 1], _inGrad[idx * 3 + 2], 0.f);
			accGrad += inGrad * wgt;

		}
	}


    _outGrad[cIdx * 3 + 0] = accGrad.x;
    _outGrad[cIdx * 3 + 1] = accGrad.y;
	_outGrad[cIdx * 3 + 2] = accGrad.z;
}

std::vector<torch::Tensor> cuda_cross_bilateral_forward(torch::Tensor input, torch::Tensor albedo, torch::Tensor normal, torch::Tensor depth, int winSize)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];
	torch::IntList kernel_shape = {height, width, 1};
	torch::Tensor outWgtSum = torch::zeros(kernel_shape, input.options());
    torch::Tensor output = torch::zeros_like(input);

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    // printf("debug bilateral\n");
	CrossBilateralFilterKernel <<<grid, threads>>> (
        input.data<float>(),
        albedo.data<float>(),
		normal.data<float>(),
        depth.data<float>(),
        output.data<float>(),
		outWgtSum.data<float>(),
        height,  // height
        width,  // width
        winSize
        );

    return {output, outWgtSum};
}



torch::Tensor cuda_cross_bilateral_backward(torch::Tensor inGrad, torch::Tensor input, torch::Tensor albedo, torch::Tensor normal, torch::Tensor depth, torch::Tensor wgtsum, int winSize)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

    // Initialize output tensor to zero
    torch::Tensor output = torch::zeros_like(input);
	
	// printf("debug\n");
	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	CrossBilateralFilterBackwardKernel <<<grid, threads>>> (
		inGrad.data<float>(),
        input.data<float>(),
        albedo.data<float>(),
		normal.data<float>(),
        depth.data<float>(),
		wgtsum.data<float>(),
        output.data<float>(),
        height,  // height
        width,  // width
        winSize
        );


    return output;
}


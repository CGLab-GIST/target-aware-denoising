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


__forceinline__ __host__ __device__ float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__forceinline__ __host__ __device__ float4 operator-(float a, float4 b) {
    return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}

__forceinline__ __host__ __device__ void operator-=(float4 &a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__forceinline__ __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__forceinline__ __host__ __device__ float4 operator/(float a, float4 b) {
    return make_float4(a / b.x, a / b.y, a / b.z,  a / b.w);
}


__forceinline__ __host__ __device__ float4 operator+(float4 a, float b) {
    return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

__forceinline__ __host__ __device__ float4 fmaxf(float4 a, float4 b) {
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x) + (a.y * a.y) + (a.z * a.z);
}


__forceinline__ __device__ float4 logTrans(const float4& a) {
	float4 outCol;
	outCol.x = __logf(a.x + 1.f);
	outCol.y = __logf(a.y + 1.f);
	outCol.z = __logf(a.z + 1.f);
	return outCol;
}



__global__ void BilateralWeightSumKernel(const float* _img, const float* _refImg, 
						  float* _X, float* _XX, float* _XY, float* _Y, float* _wgtSum, int height, int width, int winSize, float bandwidth) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;

	const float4& cRefImg = make_float4(_refImg[cIdx * 3 + 0], _refImg[cIdx * 3 + 1], _refImg[cIdx * 3 + 2], 0.f);

	int sy = max(0, cy - halfWinSize);
	int ey = min(height - 1, cy + halfWinSize);
	int sx = max(0, cx - halfWinSize);
	int ex = min(width - 1, cx + halfWinSize);

	float wgtSum = 0.f;

	float4 xSum = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 xySum = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 xxSum = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 ySum = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = sy; iy <= ey; ++iy) {
		for (int ix = sx; ix <= ex; ++ix) {
			int idx = iy * width + ix;
			const float4& iImg = make_float4(_img[idx * 3 + 0], _img[idx * 3 + 1], _img[idx * 3 + 2], 0.f);
			const float4& iRefImg = make_float4(_refImg[idx * 3 + 0], _refImg[idx * 3 + 1], _refImg[idx * 3 + 2], 0.f);
			float dist2_ref = norm2(logTrans(cRefImg) - logTrans(iRefImg)) / (2.f * bandwidth * bandwidth);
			float wgt = __expf(-dist2_ref);

			xSum += iRefImg * wgt;
			xySum += iImg * iRefImg * wgt;
			xxSum += iRefImg * iRefImg * wgt;
			ySum += iImg * wgt;
			wgtSum += wgt;
		}
	}
	float invWgtSum = 1.f / wgtSum;


	_X[cIdx * 3 + 0] = xSum.x * invWgtSum;
	_X[cIdx * 3 + 1] = xSum.y * invWgtSum;
	_X[cIdx * 3 + 2] = xSum.z * invWgtSum;

	_XY[cIdx * 3 + 0] = xySum.x * invWgtSum;
	_XY[cIdx * 3 + 1] = xySum.y * invWgtSum;
	_XY[cIdx * 3 + 2] = xySum.z * invWgtSum;

	_XX[cIdx * 3 + 0] = xxSum.x * invWgtSum;
	_XX[cIdx * 3 + 1] = xxSum.y * invWgtSum;
	_XX[cIdx * 3 + 2] = xxSum.z * invWgtSum;

	_Y[cIdx * 3 + 0] = ySum.x * invWgtSum;
	_Y[cIdx * 3 + 1] = ySum.y * invWgtSum;
	_Y[cIdx * 3 + 2] = ySum.z * invWgtSum;

	_wgtSum[cIdx] = wgtSum;

}


__global__ void BilateralWeightSumBackwardKernel(const float* _in_grad_xy, const float* _in_grad_y, const float* _img, const float* _refImg, const float* _wgtSum,		  
						  float* _gradOut, int height, int width, int winSize, float bandwidth) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;


	const float4& cRefImg = make_float4(_refImg[cIdx * 3 + 0], _refImg[cIdx * 3 + 1], _refImg[cIdx * 3 + 2], 0.f);

	int sy = max(0, cy - halfWinSize);
	int ey = min(height - 1, cy + halfWinSize);
	int sx = max(0, cx - halfWinSize);
	int ex = min(width - 1, cx + halfWinSize);

	float4 weightSumGradXY = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 weightSumGradY = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = sy; iy <= ey; ++iy) {
		for (int ix = sx; ix <= ex; ++ix) {
			int idx = iy * width + ix;
			const float4& inGradXY = make_float4(_in_grad_xy[idx * 3 + 0], _in_grad_xy[idx * 3 + 1], _in_grad_xy[idx * 3 + 2], 0.f);
			const float4& inGradY = make_float4(_in_grad_y[idx * 3 + 0], _in_grad_y[idx * 3 + 1], _in_grad_y[idx * 3 + 2], 0.f);
			const float4& iRefImg = make_float4(_refImg[idx * 3 + 0], _refImg[idx * 3 + 1], _refImg[idx * 3 + 2], 0.f);

			float dist2_ref = norm2(logTrans(cRefImg) - logTrans(iRefImg)) / (2.f * bandwidth * bandwidth);
			
			const float& wgtSum = _wgtSum[idx];

			float wgt = __expf(-dist2_ref);
			float invWgtSum = 1.f / fmaxf(FLT_EPS, wgtSum);
			wgt = wgt * invWgtSum;

			weightSumGradXY += inGradXY * wgt;
			weightSumGradY += inGradY * wgt;

		}
	}

	float4 XY_grad = weightSumGradXY * cRefImg;

	_gradOut[cIdx * 3 + 0] = XY_grad.x + weightSumGradY.x;
	_gradOut[cIdx * 3 + 1] = XY_grad.y + weightSumGradY.y;
	_gradOut[cIdx * 3 + 2] = XY_grad.z + weightSumGradY.z;
	
}


std::vector<torch::Tensor> cuda_regression_forward(torch::Tensor _rand, torch::Tensor _target, int winSize, float bandwidth)
{
    // Get tensor shapes
    torch::IntList input_shape = _rand.sizes();
	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];
	torch::IntList kernel_shape = {height, width, 1};

	torch::Tensor outWgtSum = torch::zeros(kernel_shape, _rand.options());
	torch::Tensor _XX = torch::zeros_like(_rand);
	torch::Tensor _X = torch::zeros_like(_rand);
	torch::Tensor _XY = torch::zeros_like(_rand);
	torch::Tensor _Y = torch::zeros_like(_rand);

 	// torch::Tensor _outImg = torch::zeros_like(_rand);
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	//  float timeCUDAFit = 0.f;
	//  cudaEvent_t start, stop;
	//  cudaEventCreate(&start);
	//  cudaEventCreate(&stop);
	//  cudaEventRecord(start, 0);
	

	BilateralWeightSumKernel << < grid, threads >> > (
		_rand.data<float>(),
		_target.data<float>(),
		_X.data<float>(),
		_XX.data<float>(),
		_XY.data<float>(),
		_Y.data<float>(),
		outWgtSum.data<float>(),
		height, width, winSize, bandwidth);
	

	//  cudaEventRecord(stop, 0);
	//  cudaEventSynchronize(stop);
	//  cudaEventElapsedTime(&timeCUDAFit, start, stop);	
	//  printf("INFO: CUDA running time %.1f ms\n", timeCUDAFit);


    return {_X, _XX, _XY, _Y, outWgtSum};
}

torch::Tensor cuda_regression_backward(torch::Tensor _in_grad_xy, torch::Tensor _in_grad_y, torch::Tensor _rand, torch::Tensor _target, torch::Tensor _wgtSum,
					 int winSize, float bandwidth)
{
    // Get tensor shapes
    torch::IntList input_shape = _rand.sizes();
	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];
	torch::Tensor _outGrad = torch::zeros_like(_rand);

	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	// float timeCUDAFit = 0.f;
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, 0);
	
	BilateralWeightSumBackwardKernel << < grid, threads >> > (
	_in_grad_xy.data<float>(),
	_in_grad_y.data<float>(),
	_rand.data<float>(),
	_target.data<float>(),
	_wgtSum.data<float>(),
	_outGrad.data<float>(),
	height, width, winSize, bandwidth);

	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&timeCUDAFit, start, stop);	
	// printf("INFO: CUDA running time (backward) %.1f ms\n", timeCUDAFit);


    return _outGrad;
}



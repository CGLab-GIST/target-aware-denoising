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


#define OLS_EP 0.001f
#define OLS_FEAT_DIM 9 //albedo + normal + pos + depth


float4* g_accOut = NULL;
int g_lenAccOut = 0;


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

__forceinline__ __host__ __device__ float4 fmaxf(float4 a, float4 b) {
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x) + (a.y * a.y) + (a.z * a.z);
}

__forceinline__ __device__ float4 logTrans(const float4& a) {
	float4 outCol;
	outCol.x = log(fmaxf(a.x, 0.f) + 1.f);
	outCol.y = log(fmaxf(a.y, 0.f) + 1.f);
	outCol.z = log(fmaxf(a.z, 0.f) + 1.f);
	return outCol;
}



// Input - A (only upper triangle is set!)
__device__ void cholesky(float *A, int P, float *diagL) {
	for (int i = 0; i < P; ++i) {
		for (int j = i; j < P; ++j) {
			float sum = A[i * P + j];
			for (int k = i - 1; k >= 0; --k)
				sum -= A[i * P + k] * A[j * P + k];
			if (i == j) {
				if (sum <= 0.f)
					printf("ERR in cholesky");
				diagL[i] = sqrtf(sum);
			}
			else
				A[j * P + i] = sum / diagL[i];
		}
	}
}

__device__ void cholesky_fullA(float *A, int P){

	for (int i = 0; i< P; ++ i){
		for (int j = i; j< P; ++j){
			float sum = A[i * P + j];

			for (int k = i - 1; k >= 0 ; --k){
				sum -= A[i * P + k] * A[j * P + k];
			}

			if (i == j) {
				if (sum <= 0.f)
					printf("ERR in cholesky");
				A[i * P + i] = sqrtf(sum);
			}
			else{

				A[j * P + i] = sum / A[i * P + i];
			}

		}
	}

}

__device__ void inverseMatrix(float* invA, const float* A, int P){

	for(int i = 0 ; i < P; ++i){
		for (int j = 0; j <= i; ++j){
			float sum = (i == j) ? 1.f: 0.f;
			for (int k = i - 1; k >= j; --k)
				sum -= A[i * P + k] * invA[j * P + k];
			
			invA[j * P + i] = sum / A[i * P + i];

		}
	}

	for (int i = P - 1; i >= 0; --i){
		for (int j = 0; j <= i; ++j){
			float sum = (i < j) ? 0.f: invA[j * P + i];
			for (int k = i + 1; k < P; ++k)
				sum -= A[k * P + i] * invA[j * P + k];
			
			invA[i * P + j] = invA[j * P + i] = sum / A[i * P + i];
		}
	}
}



__global__ void OlsKernel(const float* _img, const float* _depth, const float* _albedo, const float* _normal,		  
						  float* _outImg, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;

	const int P = OLS_FEAT_DIM + 1;

	float delta[P];
	float A[P * P] = { 0.f, };
	float4 XtB[P];
	for (int i = 0; i < P; ++i)
		XtB[i] = make_float4(0.f, 0.f, 0.f, 0.f);

	float bandwidth_pos = (float)(halfWinSize/3.f);
	float bandwidth = 0.1f;
	float bandwidth_alb = 0.05;
	const float& cDepth = _depth[cIdx];
	const float4& cAlbedo = make_float4(_albedo[cIdx * 3 + 0], _albedo[cIdx * 3 + 1], _albedo[cIdx * 3 + 2], 0.f);
	const float4& cNormal = make_float4(_normal[cIdx * 3 + 0], _normal[cIdx * 3 + 1], _normal[cIdx * 3 + 2], 0.f);
			

	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;

			const float4& iImg = make_float4(_img[idx * 3 + 0], _img[idx * 3 + 1], _img[idx * 3 + 2], 0.f);
			const float4& iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
			const float& iDepth = _depth[idx];

			delta[0] = 1.f;
			delta[1] = (iAlbedo.x - cAlbedo.x);
			delta[2] = (iAlbedo.y - cAlbedo.y);
			delta[3] = (iAlbedo.z - cAlbedo.z);
			delta[4] = (iNormal.x - cNormal.x);
			delta[5] = (iNormal.y - cNormal.y);
			delta[6] = (iNormal.z - cNormal.z);
			delta[7] = (iDepth - cDepth);
			delta[8] = (x - cx) /(float)halfWinSize;
			delta[9] = (y - cy) /(float)halfWinSize;

			float dist2_albedo = norm2(cAlbedo - iAlbedo) / (2.f * bandwidth_alb * bandwidth_alb);
			float dist2_normal = norm2(cNormal - iNormal) / (2.f * bandwidth * bandwidth);
			float dist2_depth = ((cDepth - iDepth) * (cDepth - iDepth)) / (2.f * bandwidth * bandwidth);
			float dist2_pos = ((cx-x)*(cx-x) + (cy-y)*(cy-y)) / (2.f * bandwidth_pos * bandwidth_pos);
			float total_dist = (dist2_albedo + dist2_normal + dist2_depth + dist2_pos);
			float weight = __expf(-total_dist);

			for (int row = 0; row < P; ++row) {
				for (int col = row; col < P; ++col) {
					A[row * P + col] += weight * delta[row] * delta[col];
				}
			}

			for (int i = 0; i < P; ++i)
				XtB[i] += weight * delta[i] * iImg;

		}
	}

	for (int row = 0; row < P; ++row)		
		A[row * P + row] += OLS_EP;

	float diagL[P];
	cholesky(A, P, diagL);

	float4 beta[P];

	for (int i = 0; i < P; ++i) {
		float4 sum = XtB[i];
		for (int k = i - 1; k >= 0; --k)
			sum = sum - A[i * P + k] * beta[k];
		beta[i] = sum / diagL[i];
	}
	// L^t \beta = y
	for (int i = P; i >= 0; --i) {
		float4 sum = beta[i];
		for (int k = i + 1; k < P; ++k)
			sum = sum - A[k * P + i] * beta[k];
		beta[i] = sum / diagL[i];
	}

	_outImg[cIdx * 3 + 0] = beta[0].x;
	_outImg[cIdx * 3 + 1] = beta[0].y;
	_outImg[cIdx * 3 + 2] = beta[0].z;
		

}



__global__ void OlsBackwardKernel(const float* _inGrad, const float* _img, const float* _depth,
								const float* _albedo, const float* _normal, 	  
						float4* _accOut,
						  int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;

	const int P = OLS_FEAT_DIM + 1;
	float delta[P];
	float A[P * P] = { 0.f, };
	float4 XtB[P];
	for (int i = 0; i < P; ++i)
		XtB[i] = make_float4(0.f, 0.f, 0.f, 0.f);
		
	float bandwidth = 0.1f;
	float bandwidth_pos = (float)(halfWinSize/3.f);
	float bandwidth_alb = 0.05;

	const float& cDepth = _depth[cIdx];
	const float4& cAlbedo = make_float4(_albedo[cIdx * 3 + 0], _albedo[cIdx * 3 + 1], _albedo[cIdx * 3 + 2], 0.f);
	const float4& cNormal = make_float4(_normal[cIdx * 3 + 0], _normal[cIdx * 3 + 1], _normal[cIdx * 3 + 2], 0.f);
	
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;

			const float4& iImg = make_float4(_img[idx * 3 + 0], _img[idx * 3 + 1], _img[idx * 3 + 2], 0.f);
			const float& iDepth = _depth[idx];
			const float4& iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
	

			delta[0] = 1.f;
			delta[1] = (iAlbedo.x - cAlbedo.x);
			delta[2] = (iAlbedo.y - cAlbedo.y);
			delta[3] = (iAlbedo.z - cAlbedo.z);
			delta[4] = (iNormal.x - cNormal.x);
			delta[5] = (iNormal.y - cNormal.y);
			delta[6] = (iNormal.z - cNormal.z);
			delta[7] = (iDepth - cDepth);
			delta[8] = (x - cx) /(float)halfWinSize;
			delta[9] = (y - cy) /(float)halfWinSize;

			float dist2_albedo = norm2(cAlbedo - iAlbedo) / (2.f * bandwidth_alb * bandwidth_alb);
			float dist2_normal = norm2(cNormal - iNormal) / (2.f * bandwidth * bandwidth);
			float dist2_depth = ((cDepth - iDepth) * (cDepth - iDepth)) / (2.f * bandwidth * bandwidth);
			float dist2_pos = ((cx-x)*(cx-x) + (cy-y)*(cy-y)) / (2.f * bandwidth_pos * bandwidth_pos);
			float total_dist = (dist2_albedo + dist2_normal + dist2_depth + dist2_pos);
			float weight = __expf(-total_dist);

			for (int row = 0; row < P; ++row) {
				for (int col = row; col < P; ++col) {
					A[row * P + col] += weight * delta[row] * delta[col];
				}
			}

			for (int i = 0; i < P; ++i)
				XtB[i] += weight * delta[i] * iImg;

		}
	}

	for (int row = 0; row < P; ++row)		
		A[row * P + row] += OLS_EP;

	float diagL[P];

	cholesky_fullA(A, P);

	float invA[P * P] = {0.f, }; 
	inverseMatrix(invA, A, P);

	float4 beta[P];
	for (int i = 0; i < P; ++i){
		float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
		for (int j = 0; j < P; ++j){
			sum += invA[i * P + j] * XtB[j];
		}
		beta[i] = sum;
	}
	const float4& cInGrad = make_float4(_inGrad[cIdx * 3 + 0], _inGrad[cIdx * 3 + 1], _inGrad[cIdx * 3 + 2], 0.f);

	float weightSum = 0.f;
	float4 sumGrad = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;

			const float4& iImg = make_float4(_img[idx * 3 + 0], _img[idx * 3 + 1], _img[idx * 3 + 2], 0.f);
			const float4& iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
			const float4& iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
			const float& iDepth = _depth[idx];

			delta[0] = 1.f;
			delta[1] = (iAlbedo.x - cAlbedo.x);
			delta[2] = (iAlbedo.y - cAlbedo.y);
			delta[3] = (iAlbedo.z - cAlbedo.z);
			delta[4] = (iNormal.x - cNormal.x);
			delta[5] = (iNormal.y - cNormal.y);
			delta[6] = (iNormal.z - cNormal.z);
			delta[7] = (iDepth - cDepth);
			delta[8] = (x - cx) / (float)halfWinSize;
			delta[9] = (y - cy) / (float)halfWinSize;

			float dist2_albedo = norm2(cAlbedo - iAlbedo) / (2.f * bandwidth_alb * bandwidth_alb);
			float dist2_normal = norm2(cNormal - iNormal) / (2.f * bandwidth * bandwidth);
			float dist2_depth = ((cDepth - iDepth) * (cDepth - iDepth)) / (2.f * bandwidth * bandwidth);
			float dist2_pos = ((cx-x)*(cx-x) + (cy-y)*(cy-y)) / (2.f * bandwidth_pos * bandwidth_pos);
			float total_dist = (dist2_albedo + dist2_normal + dist2_depth + dist2_pos);
			float weight = __expf(-total_dist);

			float sum = 0.f;
			for (int i = 0; i < P; i++){
				sum += invA[0 * P + i] * delta[i];
				
			}
			

			atomicAdd(&_accOut[idx].x, weight * sum * cInGrad.x);
			atomicAdd(&_accOut[idx].y, weight * sum * cInGrad.y);
			atomicAdd(&_accOut[idx].z, weight * sum * cInGrad.z);
			atomicAdd(&_accOut[idx].w, 1.f);
			
		}
	}
}




__global__ void OlsFinalizeKernel(float4* _accOut, float* _outCol, int height, int width) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	
	float4 accCol = _accOut[cIdx];


	_outCol[cIdx * 3 + 0] = accCol.x;
	_outCol[cIdx * 3 + 1] = accCol.y;
	_outCol[cIdx * 3 + 2] = accCol.z;

}


torch::Tensor cuda_ols_forward(torch::Tensor input, torch::Tensor depth, torch::Tensor albedo, torch::Tensor normal, int winSize)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

    // Initialize output tensor to zero
    torch::Tensor output = torch::zeros_like(input);
	
	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	OlsKernel <<<grid, threads>>> (
        input.data<float>(),
		depth.data<float>(),
		albedo.data<float>(),
		normal.data<float>(),
        output.data<float>(),
        height,  // height
        width,  // width
        winSize
        );



    return output;
}




torch::Tensor cuda_ols_backward(torch::Tensor inGrad, torch::Tensor input, torch::Tensor depth, torch::Tensor albedo, torch::Tensor normal,int winSize)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

    // Initialize output tensor to zero
    torch::Tensor output = torch::zeros_like(input);
	
	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	if (g_lenAccOut < width * height) {
		if (g_accOut)
			cudaFree(g_accOut);
		cudaError_t cudaStatus = cudaMalloc((void **)&g_accOut, width * height * sizeof(float4));
		g_lenAccOut = width * height;

		if (cudaStatus != cudaSuccess) {
			printf("Err: Malloc failed - Code: %d\n", cudaStatus);
		}
	}

	cudaMemset(g_accOut, 0, width * height * sizeof(float4));

	OlsBackwardKernel <<<grid, threads>>> (
		inGrad.data<float>(),
        input.data<float>(),
		depth.data<float>(),
		albedo.data<float>(),
		normal.data<float>(),
		g_accOut,
        height,  // height
        width,  // width
        winSize
        );


	OlsFinalizeKernel << < grid, threads >> >  (g_accOut, output.data<float>(), height, width);

    return output;
}

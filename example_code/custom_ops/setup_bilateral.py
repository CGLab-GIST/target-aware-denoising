from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CPP_FILES = [
	'cross_bilateral/cross_bilateral.cpp', 
	'cross_bilateral/cuda_cross_bilateral.cu',
]

setup(
	name='cross_bilateral',
	ext_modules=[CUDAExtension('cross_bilateral_cpp', CPP_FILES, extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["cross_bilateral/cross_bilateral"],
	cmdclass={
		'build_ext': BuildExtension
	}
)

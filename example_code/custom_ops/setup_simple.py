from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CPP_FILES = [
	'simple_regression/simple_regression.cpp', 
	'simple_regression/cuda_simple_regression.cu',
]

setup(
	name='simple_regression',
	ext_modules=[CUDAExtension('simple_regression_cpp', CPP_FILES, extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["simple_regression/simple_regression"],
	cmdclass={
		'build_ext': BuildExtension
	}
)

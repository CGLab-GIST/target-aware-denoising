from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CPP_FILES = [
	'least_squares/least_squares.cpp', 
	'least_squares/cuda_least_squares.cu',
]

setup(
	name='least_squares',
	ext_modules=[CUDAExtension('least_squares_cpp', CPP_FILES, extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["least_squares/least_squares"],
	cmdclass={
		'build_ext': BuildExtension
	},
)

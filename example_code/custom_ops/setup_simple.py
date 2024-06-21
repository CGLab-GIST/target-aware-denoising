# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CPP_FILES = [
	'simple_regression/simple_regression.cpp', 
	'simple_regression/cuda_simple_regression.cu',
]

setup(
	name='simple_regression',
	version='0.1',
	install_requires=['torch'],
	ext_modules=[CUDAExtension('simple_regression_cpp', CPP_FILES, extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["simple_regression/simple_regression"],
	cmdclass={
		'build_ext': BuildExtension
	},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
	python_requires='>=3.6',
)

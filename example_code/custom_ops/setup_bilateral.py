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
	'cross_bilateral/cross_bilateral.cpp', 
	'cross_bilateral/cuda_cross_bilateral.cu',
]

setup(
	name='cross_bilateral',
	version='0.1',
	install_requires=['torch'],
	ext_modules=[CUDAExtension('cross_bilateral_cpp', CPP_FILES, extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["cross_bilateral/cross_bilateral"],
	cmdclass={
		'build_ext': BuildExtension
	},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
	python_requires='>=3.6',
)

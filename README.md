# [Target-Aware Image Denoising for Inverse Monte Carlo Rendering](https://cglab.gist.ac.kr/sig24target/)

[Jeongmin Gu](https://jeongmingu.github.io/JeongminGu/), [Jonghee Back](https://jongheeback.notion.site/Jonghee-Back-c553120bca4144189bd9416d2fcfb0c1), [Sung-Eui Yoon](https://sgvr.kaist.ac.kr/~sungeui/), [Bochang Moon](https://cglab.gist.ac.kr/people/bochang.html)

<!-- ![Teaser](teaser.png) -->

## Overview

This repository provides the example codes for SIGGRAPH 2024 paper, [Target-Aware Image Denoising for Inverse Monte Carlo Rendering](https://cglab.gist.ac.kr/sig24target/).
You can apply our method for optimizing geometries and volumes using the specific version of Mitsuba3 as mentioned in our paper. 
For details, please refer to the paper and supplementary report on our website.


## Requirements
We recommend running this code through [Docker](https://docs.docker.com/) and [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on Ubuntu.
Please refer to the detailed instruction for the installation of [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Download test scenes

You can download the example scenes in provided codes at the link below.
In the provided code, the default path for the scene file is set to "./scenes/SCENE_NAME/scene.xml"
- [Tire](https://github.com/wchang22/ReSTIR_DR)
- [Veach-ajar](https://benedikt-bitterli.me/resources/)
- [Curtain](https://cglab.gist.ac.kr/resources/)  
In the paper, we use a Mars image (i.e., albedo textures) as the initial parameters, which you can download from this [website](https://www.solarsystemscope.com/textures/).
After that, please copy the downloaded textures in "./scenes/Curtain/textures/2k_mars.jpg".

## Test with example codes
We provide the codes for various denoisers (e.g., cross-bilateral, [OIDN](https://www.openimagedenoise.org/), linear regression with G-buffers) and our target-aware denoiser for inverse MC rendering.
For running the provided codes, you can proceed in the following order:

1. Build and run docker image 
```
bash run_docker.sh
```
2. Build PyTorch custom operators (CUDA) 
```
cd custom_ops
python setup.py install
python setup_bilateral.py install
python setup_simple.py install
```
3. Run the script
```
bash run_mts.sh
```

<!-- ## License

All source codes are released under a [BSD License](license). -->


<!-- ## Citation

```
@article{10.1145/3550454.3555496,
author = {Gu, Jeongmin and Iglesias-Guitian, Jose A. and Moon, Bochang},
title = {Neural James-Stein Combiner for Unbiased and Biased Renderings},
year = {2022},
issue_date = {December 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3550454.3555496},
doi = {10.1145/3550454.3555496},
journal = {ACM Trans. Graph.},
month = {nov},
articleno = {262},
numpages = {14},
keywords = {james-stein estimator, learning-based denoising, james-stein combiner, monte carlo rendering}
}

``` -->

## Contact

If there are any questions, issues or comments, please feel free to send an e-mail to [jeong755@gm.gist.ac.kr](mailto:jeong755@gm.gist.ac.kr).



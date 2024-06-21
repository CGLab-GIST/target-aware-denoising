


# cd ./custom_ops && python setup.py install && python setup_bilateral.py install && python setup_simple.py install && cd ..


### RUN baseline
# python example_ops.py --scene veach-ajar --loss L1 --spp 16 --backward_spp 16 --iter 400
# python example_ops.py --scene tire --loss L1 --spp 8 --backward_spp 16 --iter 400
# python example_ops.py --scene curtain --loss L1 --spp 128 --backward_spp 128 --iter 400


### RUN Ours
# python example_ops.py --scene veach-ajar --loss L1 --spp 16 --backward_spp 16 --iter 400 --denoise_ours
# python example_ops.py --scene tire --loss L1 --spp 8 --backward_spp 16 --iter 400 --denoise_ours
# python example_ops.py --scene curtain --loss L1 --spp 128 --backward_spp 128 --iter 400 --denoise_ours

### RUN cross-bilateral
# python example_ops.py --scene tire --loss L1 --spp 8 --backward_spp 16 --iter 400 --denoise
# python example_ops.py --scene curtain --loss L1 --spp 128 --backward_spp 128 --iter 400 --denoise
# python example_ops.py --scene veach-ajar --loss L1 --spp 16 --backward_spp 16 --iter 400 --denoise

### RUN OIDN
# python example_ops.py --scene tire --loss L1 --spp 8 --backward_spp 16 --iter 400 --denoise_oidn
# python example_ops.py --scene curtain --loss L1 --spp 128 --backward_spp 128 --iter 400 --denoise_oidn
# python example_ops.py --scene veach-ajar --loss L1 --spp 16 --backward_spp 16 --iter 400 --denoise_oidn

### RUN Reg. /w G-buffers
python example_ops.py --scene tire --loss L1 --spp 8 --backward_spp 16 --iter 400 --denoise_wls
# python example_ops.py --scene curtain --loss L1 --spp 128 --backward_spp 128 --iter 400 --denoise_wls
# python example_ops.py --scene veach-ajar --loss L1 --spp 16 --backward_spp 16 --iter 400 --denoise_wls




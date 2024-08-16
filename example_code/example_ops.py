#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import mitsuba as mi
import drjit as dr
import os
import torch_ops
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="tire", help='veach-ajar, tire, curtain')
parser.add_argument("--iter", type=int, default=400)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--bandwidth", type=float, default=0.1)
parser.add_argument("--spp", type=int, default=16)
parser.add_argument("--backward_spp", type=int, default=16)
parser.add_argument("--winSize", type=int, default=31)
parser.add_argument("--loss", type=str, choices=['L1', 'L2', 'RelativeL2'], default="L1")
parser.add_argument("--denoise_ours", action="store_true")
parser.add_argument("--denoise_ours_temporal", action="store_true")
parser.add_argument("--denoise", action="store_true")
parser.add_argument("--denoise_wls", action="store_true")
parser.add_argument("--denoise_oidn", action="store_true")

args, unknown = parser.parse_known_args()

# mi.set_variant('llvm_ad_rgb')
mi.set_variant('cuda_ad_rgb')

###define custom ops
from custom_ops.simple_regression import Regression
from custom_ops.least_squares import LeastSquares
from custom_ops.cross_bilateral import GbufferCrossBilateral

base_cross_bilateral = GbufferCrossBilateral(winSize=args.winSize)
regression = Regression(win_size=args.winSize, bandwidth=args.bandwidth)
least_squares = LeastSquares(win_size=args.winSize)


### Set global parameters
ITERATION           = args.iter


if args.scene == "tire":
    scene_dir = './scenes/Tire'
    scene_name = 'scene.xml'
    key = 'mat-tire.brdf_0.roughness.data'
    resX = 1024
    resY = 1024
    TRAIN_SPP          = args.spp
    TRAIN_BACKWARD_SPP = args.backward_spp
    TEST_SPP = 512
    VALID_SPP = 512
    MAX_DEPTH = 8

elif args.scene == "veach-ajar":
    scene_dir = './scenes/Veach-ajar'
    scene_name = 'scene.xml'
    key = 'LandscapeBSDF.brdf_0.reflectance.data'
    resX = 1280
    resY = 720
    TRAIN_SPP           = args.spp
    TRAIN_BACKWARD_SPP  = args.backward_spp
    TARGET_SPP = 16384 * 8
    TEST_SPP = TARGET_SPP
    VALID_SPP = 1024
    MAX_DEPTH = 13

elif args.scene == "curtain":
    scene_dir = './scenes/Curtain'
    scene_name = 'scene.xml'
    key = 'BC_sphere.reflectance.data'
    resX = 1024
    resY = 1024
    TRAIN_SPP           = args.spp
    TRAIN_BACKWARD_SPP  = args.backward_spp
    TARGET_SPP = 16384 * 4
    TEST_SPP = TARGET_SPP
    MAX_DEPTH = 64
    VALID_SPP = 1024


def render_gt(scene, render_spp):
    num_render = int(render_spp / 512)
    image = dr.zeros(mi.TensorXf, (resY, resX, 3))
    for i in range(0, num_render):
        image_sub = mi.render(scene, spp=512, seed = i)
        image += image_sub
    image /= num_render
    return image

def get_elapsed_execution_time():
    ### We use the same function that was provided by parameter-space ReSTIR [chang et al. 2023] 
    hist = dr.kernel_history()
    elapsed_time = 0
    for entry in hist:
        elapsed_time += entry['execution_time']
    return elapsed_time


### set target parameter
featUsed = args.denoise or args.denoise_wls or args.denoise_oidn

scene_path = os.path.join(scene_dir, scene_name)
scene = mi.load_file(scene_path, integrator='prb', resx= resX, resy=resY, max_depth = MAX_DEPTH)

prb_integrator = mi.load_dict({'type': 'prb', 'max_depth': MAX_DEPTH})


aov_integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'albedo:albedo, normals:sh_normal, dd.y:depth'
        })


params = mi.traverse(scene)
param_ref = mi.TensorXf(params[key])
param_shape = np.array(params[key].shape)

### render a  reference image
gt_path = os.path.join(scene_dir, 'target.exr')
if not os.path.exists(gt_path):
    print("[RENDER GT]")
    image_ref = render_gt(scene, TARGET_SPP)
    mi.util.write_bitmap(os.path.join(scene_dir, 'target.exr'), image_ref)
else:
    image_ref = mi.Bitmap(gt_path).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
    image_ref = mi.TensorXf(image_ref)


def train(spp_forward, spp_backward, optimizer = 'Adam'):

    ### set optimizer for Mitsuba
    if optimizer == 'SGD':
        opt = mi.ad.SGD(lr=args.lr)
    else:
        opt = mi.ad.Adam(lr = args.lr)
        
    ## initialize the target parameter and render 
    param_initial = np.full(param_shape.tolist(), 0.5)

    """
     In our paper, we use a Mars image as the initial parameters for the Curtain scene,
     and the Mars texture was provided by Solar System Scope: https://www.solarsystemscope.com/textures/
     Note that the texture resolution should be the same as the target texture resolution (e.g., 1024x512). 
    """
    if (args.scene == "curtain"):
        param_initial = mi.Bitmap(os.path.join(scene_dir, "textures","2k_mars.jpg")).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
        param_initial = np.resize(param_initial, (param_shape[0], param_shape[1], param_shape[2]))
        
    params[key] = mi.TensorXf(param_initial)
    params.update();

    opt[key] = params[key] 
    params.update(opt);
    scene.integrator().param_name = key
    dr.set_flag(dr.JitFlag.KernelHistory, 1) 

    np.random.seed(0)

    losses = []
    iterations = []
    total_time = 0
    total_time_list = []
    for it in range(ITERATION):
        seed_f = np.random.randint(2**31)
        seed_b = np.random.randint(2**31)                                                                                                                                                                                                            
        image = mi.render(scene, params, integrator = prb_integrator, spp=spp_forward,  spp_grad = spp_backward, seed = seed_f, seed_grad=seed_b)

        if featUsed:
            ### We ignore the gradients of G-buffers, as discussed in supplemental report
            with dr.suspend_grad():
                aovs = mi.render(scene, params, integrator = aov_integrator, spp=spp_forward,  spp_grad = spp_backward, seed = seed_f, seed_grad=seed_b)
                albedo = aovs[:,:,0:3]
                normal = aovs[:,:,3:6]
                depth = aovs[:,:,6:7]
                albedo = dr.clamp(albedo, 0.0, 1.0)
                max_depth = dr.max(depth)
                depth /= max_depth
            
        if args.denoise:
            image= torch_ops.base_cross_bilateral_op(base_cross_bilateral, image, albedo, normal, depth)
        elif args.denoise_wls:
            image = torch_ops.least_squares_op(least_squares, image, depth, albedo, normal)
        elif args.denoise_ours:
            image = torch_ops.weighted_simple_regression_op(regression, image, image_ref)
        elif args.denoise_oidn:
            image = torch_ops.oidn_op(image, albedo, normal)


        if (args.loss == "RelativeL2"):
            pixel_loss = torch_ops.rel_l2_loss_op(image, image_ref)
        elif (args.loss == "L2"):
            pixel_loss = dr.sqr(image - image_ref)
        else:
            pixel_loss = dr.abs(image - image_ref)
        

        loss = dr.mean(pixel_loss)
        dr.backward(loss)

        # grad = dr.grad(opt[key])

        opt.step()
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)
        params.update(opt)

        total_time += get_elapsed_execution_time()
        
        total_time_list.append(total_time)
        losses.append(loss[0])

            
        iterations.append(it)

        print(f'-- Iteration {it} -- Loss {losses[-1]:.4f}')
        it += 1

    print('\nOptimization complete.')
    print('\nTotal Training Time: %.4f.'%total_time)
    return iterations, losses, mi.TensorXf(params[key]), total_time_list



iterations, total_losses, updated_param, total_time_list = train(TRAIN_SPP, TRAIN_BACKWARD_SPP)

params[key] = updated_param
params.update();

image_final = render_gt(scene, TEST_SPP)

if args.denoise_ours:
    mi.util.write_bitmap(os.path.join(scene_dir, 'render_final_ours_%s_iter%d_lr%d_%dspp_backward_%dspp.exr'%(args.loss, ITERATION, int(args.lr * 100), TRAIN_SPP, TRAIN_BACKWARD_SPP)), image_final)
elif args.denoise:
    mi.util.write_bitmap(os.path.join(scene_dir, 'render_final_cb_%s_iter%d_lr%d_%dspp_backward_%dspp.exr'%(args.loss, ITERATION, int(args.lr * 100), TRAIN_SPP, TRAIN_BACKWARD_SPP)), image_final)
elif args.denoise_wls:
    mi.util.write_bitmap(os.path.join(scene_dir, 'render_final_wls_%s_iter%d_lr%d_%dspp_backward_%dspp.exr'%(args.loss, ITERATION, int(args.lr * 100), TRAIN_SPP, TRAIN_BACKWARD_SPP)), image_final)
elif args.denoise_oidn:
    mi.util.write_bitmap(os.path.join(scene_dir, 'render_final_oidn_%s_iter%d_lr%d_%dspp_backward_%dspp.exr'%(args.loss, ITERATION, int(args.lr * 100), TRAIN_SPP, TRAIN_BACKWARD_SPP)), image_final)
else:
    mi.util.write_bitmap(os.path.join(scene_dir, 'render_final_%s_iter%d_lr%d_%dspp_backward_%dspp.exr'%(args.loss, ITERATION, int(args.lr * 100), TRAIN_SPP,TRAIN_BACKWARD_SPP)), image_final)

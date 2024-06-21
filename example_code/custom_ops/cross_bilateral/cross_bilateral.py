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

import torch
import cross_bilateral_cpp

class CrossBilateralFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, albedo, normal, depth, winSize):
		assert not hasattr(ctx, 'win_size') or ctx.win_size is None
		ctx.winSize = winSize
		output, outWgtsum = cross_bilateral_cpp.cross_bilateral_forward(input, albedo, normal, depth, winSize)
		ctx.save_for_backward(input, albedo, normal, depth, outWgtsum)
		return output
	
	@staticmethod
	def backward(ctx, grad_out):
		grad_input = cross_bilateral_cpp.cross_bilateral_backward(grad_out.contiguous(), *ctx.saved_variables, ctx.winSize)
		return grad_input, None, None, None, None, None

class GbufferCrossBilateral(torch.nn.Module):
	def __init__(self, winSize):
		super(GbufferCrossBilateral, self).__init__()
		self.winSize  = winSize
	def forward(self, input, albedo, normal, depth):
		output = CrossBilateralFunction.apply(input, albedo, normal, depth, self.winSize)
		return output
	


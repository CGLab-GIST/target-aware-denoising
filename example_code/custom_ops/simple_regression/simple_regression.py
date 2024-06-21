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
import simple_regression_cpp

class Regressionfunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, rand, target, winSize, bandwidth):
		x, xx, xy, y, wgtSum = simple_regression_cpp.regression_forward(rand, target, winSize, bandwidth)
		ctx._win_size = winSize
		ctx._bandwidth = bandwidth
		ctx.save_for_backward(rand, target, wgtSum)
		return x, xx, xy, y
	@staticmethod
	def backward(ctx, in_grad_x, in_grad_xx, in_grad_xy, in_grad_y):
		grad_input = simple_regression_cpp.regression_backward(in_grad_xy.contiguous(), in_grad_y.contiguous(), *ctx.saved_variables, ctx._win_size, ctx._bandwidth)
		return grad_input, None, None, None, None, None
		
class Regression(torch.nn.Module):
	def __init__(self, win_size, bandwidth):
		super(Regression, self).__init__()
		self.winSize = win_size
		self.bandwidth = bandwidth
	def forward(self, rand, target):
		x, xx, xy, y = Regressionfunction.apply(rand, target, self.winSize, self.bandwidth)
		return x, xx, xy, y


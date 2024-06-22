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
import least_squares_cpp

class LeastSquaresFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, depth, albedo, normal, win_size):
		ctx.save_for_backward(input, depth, albedo, normal)
		assert not hasattr(ctx, 'win_size') or ctx.win_size is None
		ctx._win_size = win_size
		return least_squares_cpp.least_squares_forward(input, depth, albedo, normal, win_size)
	
	def backward(ctx, grad_out):
		grad_input = least_squares_cpp.least_squares_backward(grad_out.contiguous(), *ctx.saved_variables, ctx._win_size)
		return grad_input, None, None, None, None, None

class LeastSquares(torch.nn.Module):
	def __init__(self,win_size):
		super(LeastSquares, self).__init__()
		self.win_size  = win_size
	def forward(self, input, depth, albedo, normal):
		output = LeastSquaresFunction.apply(input, depth, albedo, normal, self.win_size)
		return output

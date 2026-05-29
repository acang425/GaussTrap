import numpy as np
import torch
import torch.nn as nn


class GN(nn.Module):

	def __init__(self, var=0.01, mean=0):
		super(GN, self).__init__()
		self.var = var
		self.mean = mean

	def gaussian_noise(self, pc, mean, var):
		noise = torch.Tensor(np.random.normal(mean, var, pc.shape)).to(pc.device)
		out = pc + noise
		return out

	def forward(self, gaussians):
		new_xyz = self.gaussian_noise(gaussians._xyz, self.mean, self.var)
		return {"xyz": new_xyz, "opacity": gaussians._opacity, "features_dc": gaussians._features_dc, 
			"features_rest": gaussians._features_rest, "scaling": gaussians._scaling, "rotation": gaussians._rotation}
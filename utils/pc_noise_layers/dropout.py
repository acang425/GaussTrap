import torch
import torch.nn as nn
import numpy as np
import random

class Dropout(nn.Module):
    """
    Drops random points from the point cloud
    """
    def __init__(self, drop_rate=0.1):
        super(Dropout, self).__init__()
        self.drop_rate = drop_rate


    def forward(self, gaussians):

        mask = torch.ones(gaussians._xyz.shape[0])
        mask = mask > 0
        drop_num = int(gaussians._xyz.shape[0] * self.drop_rate)
        drop_list = random.sample(range(0, gaussians._xyz.shape[0]), drop_num)

        for index in drop_list:
            mask[index] = False

        new_lenth = gaussians._xyz[:, 0][mask].shape[0]
        new_xyz = torch.zeros([new_lenth, 3])

        new_xyz[:, 0] = gaussians._xyz[:, 0][mask]
        new_xyz[:, 1] = gaussians._xyz[:, 1][mask]
        new_xyz[:, 2] = gaussians._xyz[:, 2][mask]

        new_xyz = new_xyz.cuda()
        
        new_opacity = gaussians._opacity[mask]

        new_features_dc = gaussians._features_dc[mask]

        new_features_rest = gaussians._features_rest[mask]

        new_scaling = gaussians._scaling[mask]

        new_rotation = gaussians._rotation[mask]

        return {"xyz": new_xyz, "opacity": new_opacity, "features_dc": new_features_dc, "features_rest": new_features_rest, "scaling": new_scaling, "rotation": new_rotation}
        
        # return {"xyz": gaussians._xyz, "opacity": gaussians._opacity, "features_dc": gaussians._features_dc, 
        #         "features_rest": gaussians._features_rest, "scaling": gaussians._scaling, "rotation": gaussians._rotation}


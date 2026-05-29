import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, gaussians):
        return {"xyz": gaussians._xyz, "opacity": gaussians._opacity, "features_dc": gaussians._features_dc, 
                "features_rest": gaussians._features_rest, "scaling": gaussians._scaling, "rotation": gaussians._rotation}

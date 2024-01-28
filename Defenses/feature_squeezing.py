import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class FeatureSqueezer(torch.nn.Module):
    def __init__(self, bit_depth=1):
        super(FeatureSqueezer, self).__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        # Reduce color depth
        x = self.reduce_color_depth(x)
        # Apply spatial smoothing (e.g., median filter)
        x = self.spatial_smoothing(x)
        return x

    def reduce_color_depth(self, x):
        scale = 255 / (2 ** self.bit_depth - 1)
        x = (x * scale).int().float() / scale
        return x

    def spatial_smoothing(self, x):
        kernel_size = 3
        padding = kernel_size // 2
        x = x.unsqueeze(1) 
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        x = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x.squeeze(1)  

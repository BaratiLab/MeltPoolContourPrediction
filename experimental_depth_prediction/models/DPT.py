from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation
import torch.nn as nn
class DPTFineTune(nn.Module):
    def __init__(self):
        super(DPTFineTune, self).__init__()
        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"])
        config = DPTConfig(backbone_config=backbone_config)

        self.dpt = DPTForSemanticSegmentation(config=config)
    def forward(self, x):
       
        return self.dpt(x).logits
import torch
from torch import nn

from models.Temporal_Transformer import Temporal_Transformer_Cls
from models.Temporal_Transformer import Temporal_Transformer_DoubleCls
from models.Temporal_Transformer import Temporal_Transformer_Mean


class GenerateModel(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.args = args
        self.dtype = torch.float32
        self.image_encoder = clip_model.visual
        if args.temporal_net == "transformer":
            if args.cls_type == "mean":
                self.temporal_net = Temporal_Transformer_Mean(
                    num_patches=16,
                    input_dim=512,
                    depth=args.temporal_layers,
                    heads=8,
                    mlp_dim=1024,
                    dim_head=64,
                )
            elif args.cls_type == "cls":
                self.temporal_net = Temporal_Transformer_Cls(
                    num_patches=16,
                    input_dim=512,
                    depth=args.temporal_layers,
                    heads=8,
                    mlp_dim=1024,
                    dim_head=64,
                )
            elif args.cls_type == "double":
                self.temporal_net = Temporal_Transformer_DoubleCls(
                    num_patches=16,
                    input_dim=512,
                    depth=args.temporal_layers,
                    heads=8,
                    mlp_dim=1024,
                    dim_head=64,
                )
        self.fc = nn.Linear(512, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, image):
        n, t, c, h, w = image.shape
        image = image.contiguous().view(-1, c, h, w)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features.contiguous().view(n, t, -1)
        video_features = self.temporal_net(image_features)
        output = self.fc(video_features)
        output = self.final_activation(output)
        return output

# ML
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# SOD
from attention import Aggregation, ObjectAttention, RFBBlock
from effnet import EfficientNet, get_model_shape


class TRACER(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.block_idx, self.channels = get_model_shape(int(cfg.arch))
        self.model = EfficientNet.from_pretrained(
            f"efficientnet-b{cfg.arch}", advprop=True, cfg=cfg
        )

        # Receptive Field Blocks
        channels = [int(arg_c) for arg_c in cfg.RFB_aggregated_channel]
        self.rfb2 = RFBBlock(self.channels[1], channels[0])
        self.rfb3 = RFBBlock(self.channels[2], channels[1])
        self.rfb4 = RFBBlock(self.channels[3], channels[2])

        # Multi-level aggregation
        self.agg = Aggregation(channels)

        # Object Attention
        self.ObjectAttention2 = ObjectAttention(channel=self.channels[1], kernel_size=3)
        self.ObjectAttention1 = ObjectAttention(channel=self.channels[0], kernel_size=3)

    def forward(self, inputs):
        B, C, H, W = inputs.size()

        # EfficientNet backbone Encoder
        x = self.model.initial_conv(inputs)
        features, edge = self.model.get_blocks(x, H, W)

        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)

        ds_map0 = F.interpolate(D_0, scale_factor=8, mode="bilinear")

        D_1 = self.ObjectAttention2(D_0, features[1])
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode="bilinear")

        ds_map = F.interpolate(D_1, scale_factor=2, mode="bilinear")
        D_2 = self.ObjectAttention1(ds_map, features[0])
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode="bilinear")

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3
        return (
            torch.sigmoid(final_map),
            torch.sigmoid(edge),
            (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2)),
        )


model_info = pd.DataFrame(
    {
        "Model": ["TE0", "TE1", "TE2", "TE3", "TE4", "TE5", "TE6", "TE7"],
        "#Params": [
            "7.45M",
            "9.96M",
            "11.09M",
            "14.02M",
            "20.71M",
            "31.3M",
            "43.47M",
            "66.27M",
        ],
        "Img size": [320, 320, 352, 384, 448, 512, 576, 640],
    }
)

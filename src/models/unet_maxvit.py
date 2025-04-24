import timm
import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    def __init__(self, backbone_name="maxvit_small_tf_512.in1k", pretrained=True, num_classes=3, input_channels=12):
        super().__init__()
        # エンコーダ（特徴マップ抽出）
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, in_chans=input_channels
        )

        # 各エンコーダ出力のチャンネル数
        channels = [f["num_chs"] for f in self.encoder.feature_info]  # 例: [64, 96, 192, 384, 768]

        # Center (最深部の特徴量に1回Conv)
        self.center = nn.Conv2d(channels[-1], 512, kernel_size=3, padding=1)

        # デコーダ
        self.decoder4 = self._decoder_block(512 + channels[-2], 256)
        self.decoder3 = self._decoder_block(256 + channels[-3], 128)
        self.decoder2 = self._decoder_block(128 + channels[-4], 64)
        self.decoder1 = self._decoder_block(64 + channels[-5], 32)

        # 最終出力層
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder features
        enc_feats = self.encoder(x)

        # Center block
        x = self.center(enc_feats[-1])

        # Decoder blocks with skip connections
        x = self.decoder4(
            torch.cat(
                [F.interpolate(x, size=enc_feats[-2].shape[2:], mode="bilinear", align_corners=False), enc_feats[-2]],
                dim=1,
            )
        )

        x = self.decoder3(
            torch.cat(
                [F.interpolate(x, size=enc_feats[-3].shape[2:], mode="bilinear", align_corners=False), enc_feats[-3]],
                dim=1,
            )
        )

        x = self.decoder2(
            torch.cat(
                [F.interpolate(x, size=enc_feats[-4].shape[2:], mode="bilinear", align_corners=False), enc_feats[-4]],
                dim=1,
            )
        )

        x = self.decoder1(
            torch.cat(
                [F.interpolate(x, size=enc_feats[-5].shape[2:], mode="bilinear", align_corners=False), enc_feats[-5]],
                dim=1,
            )
        )

        return self.final(x)
